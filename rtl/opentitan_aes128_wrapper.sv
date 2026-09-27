// Deterministic V2 wrapper around OpenTitan aes_cipher_core.
//
// Correct protocol:
//   - aes_cipher_core expects in_valid_i and crypt_i together while IDLE.
//   - state_o is the direct last-round output and must be captured while
//     out_valid_o is asserted.
//
// No fault injection is performed here.

module opentitan_aes128_wrapper
  import aes_pkg::*;
(
  input  logic         clk_i,
  input  logic         rst_ni,
  input  logic         start_i,
  input  logic [127:0] plaintext_i,
  input  logic [127:0] key_i,

  output logic         ready_o,
  output logic         busy_o,
  output logic         done_o,
  output logic [127:0] ciphertext_o,
  output logic         alert_o
);

  localparam int NumShares = 1;

  typedef enum logic [1:0] {
    WrapIdle,
    WrapLaunch,
    WrapWait
  } wrap_state_e;

  wrap_state_e wrap_state_q, wrap_state_d;

  logic [127:0] plaintext_q;
  logic [127:0] key_q;
  logic [127:0] ciphertext_q;

  sp2v_e in_valid;
  sp2v_e in_ready;
  sp2v_e out_valid;
  sp2v_e out_ready;
  sp2v_e crypt_req;
  sp2v_e crypt_busy;

  sp2v_e dec_key_gen_busy;
  logic  prng_reseed_busy;
  logic  key_clear_busy;
  logic  data_out_clear_busy;

  logic [3:0][3:0][7:0] state_init [NumShares];
  logic [3:0][3:0][7:0] state_out  [NumShares];
  logic [7:0][31:0] key_init [NumShares];

  logic [3:0][31:0] plaintext_words;
  logic [3:0][31:0] ciphertext_words_now;

  logic [3:0][3:0][7:0] unused_data_in_mask;
  logic unused_entropy_req;

  function automatic logic [31:0] rev32(input logic [31:0] x);
    rev32 = {x[7:0], x[15:8], x[23:16], x[31:24]};
  endfunction

  // Match OpenTitan aes_core's data path:
  // CSR-style 32-bit words -> aes_transpose() -> cipher state.
  always_comb begin
    plaintext_words[0] = rev32(plaintext_q[127:96]);
    plaintext_words[1] = rev32(plaintext_q[95:64]);
    plaintext_words[2] = rev32(plaintext_q[63:32]);
    plaintext_words[3] = rev32(plaintext_q[31:0]);

    state_init[0] = aes_transpose(plaintext_words);

    // OpenTitan key_init is an array of 32-bit key words.
    key_init[0] = '0;
    key_init[0][0] = rev32(key_q[127:96]);
    key_init[0][1] = rev32(key_q[95:64]);
    key_init[0][2] = rev32(key_q[63:32]);
    key_init[0][3] = rev32(key_q[31:0]);

    // OpenTitan aes_core converts the final cipher state back using
    // aes_transpose(). state_out is only guaranteed valid with out_valid.
    ciphertext_words_now = aes_transpose(state_out[0]);
  end

  function automatic logic [127:0] words_to_external(
    input logic [3:0][31:0] words
  );
    words_to_external = {
      rev32(words[0]),
      rev32(words[1]),
      rev32(words[2]),
      rev32(words[3])
    };
  endfunction

  assign ciphertext_o = ciphertext_q;

  // The cipher FSM expects the input-valid handshake and crypt command in the
  // SAME IDLE transaction.
  assign in_valid  = (wrap_state_q == WrapLaunch) ? SP2V_HIGH : SP2V_LOW;
  assign crypt_req = (wrap_state_q == WrapLaunch) ? SP2V_HIGH : SP2V_LOW;
  assign out_ready = SP2V_HIGH;

  assign ready_o = (wrap_state_q == WrapIdle);
  assign busy_o  = (wrap_state_q != WrapIdle);

  always_comb begin
    wrap_state_d = wrap_state_q;

    unique case (wrap_state_q)
      WrapIdle: begin
        if (start_i) begin
          wrap_state_d = WrapLaunch;
        end
      end

      WrapLaunch: begin
        // Hold both request signals until the core accepts them.
        if (in_ready == SP2V_HIGH) begin
          wrap_state_d = WrapWait;
        end
      end

      WrapWait: begin
        if (out_valid == SP2V_HIGH) begin
          wrap_state_d = WrapIdle;
        end
      end

      default: wrap_state_d = WrapIdle;
    endcase
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wrap_state_q <= WrapIdle;
      plaintext_q  <= '0;
      key_q        <= '0;
      ciphertext_q <= '0;
      done_o       <= 1'b0;
    end else begin
      wrap_state_q <= wrap_state_d;
      done_o       <= 1'b0;

      if ((wrap_state_q == WrapIdle) && start_i) begin
        plaintext_q <= plaintext_i;
        key_q       <= key_i;
      end

      // CRITICAL: state_o is not stored by aes_cipher_core.
      // Capture the result on the exact cycle out_valid is asserted.
      if (out_valid == SP2V_HIGH) begin
        ciphertext_q <= words_to_external(ciphertext_words_now);
        done_o       <= 1'b1;
      end
    end
  end

  aes_cipher_core #(
    .AES192Enable         (1'b0),
    .CiphOpFwdOnly        (1'b1),
    .SecMasking           (1'b0),
    .SecSBoxImpl          (SBoxImplCanright),
    .SecAllowForcingMasks (1'b0),
    .SecSkipPRNGReseeding (1'b1)
  ) u_aes_cipher_core (
    .clk_i                (clk_i),
    .rst_ni               (rst_ni),

    .in_valid_i           (in_valid),
    .in_ready_o           (in_ready),

    .out_valid_o          (out_valid),
    .out_ready_i          (out_ready),

    .cfg_valid_i          (1'b1),
    .op_i                 (CIPH_FWD),
    .key_len_i            (AES_128),

    .crypt_i              (crypt_req),
    .crypt_o              (crypt_busy),

    .dec_key_gen_i        (SP2V_LOW),
    .dec_key_gen_o        (dec_key_gen_busy),

    .prng_reseed_i        (1'b0),
    .prng_reseed_o        (prng_reseed_busy),

    .key_clear_i          (1'b0),
    .key_clear_o          (key_clear_busy),

    .data_out_clear_i     (1'b0),
    .data_out_clear_o     (data_out_clear_busy),

    .alert_fatal_i        (1'b0),
    .alert_o              (alert_o),

    .prd_clearing_state_i ('{default: '0}),
    .prd_clearing_key_i   ('{default: '0}),

    .force_masks_i        (1'b0),
    .data_in_mask_o       (unused_data_in_mask),

    .entropy_req_o        (unused_entropy_req),
    .entropy_ack_i        (1'b0),
    .entropy_i            ('0),

    .state_init_i         (state_init),
    .key_init_i           (key_init),
    .state_o              (state_out)
  );

  logic unused_status;
  assign unused_status = ^{
    crypt_busy,
    dec_key_gen_busy,
    prng_reseed_busy,
    key_clear_busy,
    data_out_clear_busy,
    unused_entropy_req,
    unused_data_in_mask
  };

endmodule
