module opentitan_hmac_sha256_bridge
  import prim_sha2_pkg::*;
(
    input  logic clk_i,
    input  logic rst_ni,

    // --------------------------------------------------------
    // Core-native secret-key representation.
    // Final user-friendly 256-bit key packing comes later.
    // --------------------------------------------------------
    input  logic [1023:0] secret_key_i,

    // --------------------------------------------------------
    // HMAC transaction controls
    // --------------------------------------------------------
    input  logic reg_hash_start_i,
    input  logic reg_hash_stop_i,
    input  logic reg_hash_continue_i,
    input  logic reg_hash_process_i,

    // --------------------------------------------------------
    // External message FIFO side
    // --------------------------------------------------------
    input  logic        fifo_rvalid_i,
    input  logic [31:0] fifo_rdata_i,
    input  logic [3:0]  fifo_rmask_i,
    output logic        fifo_rready_o,

    // --------------------------------------------------------
    // HMAC-generated FIFO write control.
    // Needed for inner-digest -> outer-HMAC round.
    // --------------------------------------------------------
    output logic       fifo_wsel_o,
    output logic       fifo_wvalid_o,
    output logic [3:0] fifo_wdata_sel_o,
    input  logic       fifo_wready_i,

    // Message length in bits.
    input logic [63:0] message_length_i,

    // --------------------------------------------------------
    // Results/status
    // --------------------------------------------------------
    output logic         hash_done_o,
    output logic         hmac_idle_o,
    output logic         sha_idle_o,
    output logic         hash_running_o,
    output logic         digest_on_blk_o,
    output logic [255:0] digest_o
);

    // ========================================================
    // Fixed configuration for this project stage
    // ========================================================

    digest_mode_e digest_size;
    key_length_e  key_length;

    assign digest_size = SHA2_256;
    assign key_length  = Key_256;


    // ========================================================
    // Message FIFO structure
    // ========================================================

    sha_fifo32_t fifo_rdata;

    assign fifo_rdata.data = fifo_rdata_i;
    assign fifo_rdata.mask = fifo_rmask_i;


    // ========================================================
    // HMAC <-> SHA internal interface
    // ========================================================

    logic        sha_hash_start;
    logic        sha_hash_continue;
    logic        sha_hash_process;
    logic        sha_hash_done;

    logic        shaf_rvalid;
    sha_fifo32_t shaf_rdata;
    logic        shaf_rready;

    logic [63:0] sha_message_length;


    // ========================================================
    // SHA digest state
    // ========================================================

    sha_word64_t [7:0] digest;
    sha_word64_t [7:0] digest_initial;
    logic [7:0]         digest_we;

    assign digest_initial = '0;
    assign digest_we      = '0;


    // ========================================================
    // OpenTitan HMAC core
    // ========================================================

    hmac_core u_hmac_core (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),

        .secret_key_i             (secret_key_i),
        .hmac_en_i                (1'b1),
        .digest_size_i            (digest_size),
        .key_length_i             (key_length),

        .reg_hash_start_i         (reg_hash_start_i),
        .reg_hash_stop_i          (reg_hash_stop_i),
        .reg_hash_continue_i      (reg_hash_continue_i),
        .reg_hash_process_i       (reg_hash_process_i),

        .hash_done_o              (hash_done_o),

        .sha_hash_start_o         (sha_hash_start),
        .sha_hash_continue_o      (sha_hash_continue),
        .sha_hash_process_o       (sha_hash_process),
        .sha_hash_done_i          (sha_hash_done),

        .sha_rvalid_o             (shaf_rvalid),
        .sha_rdata_o              (shaf_rdata),
        .sha_rready_i             (shaf_rready),

        .fifo_rvalid_i            (fifo_rvalid_i),
        .fifo_rdata_i             (fifo_rdata),
        .fifo_rready_o            (fifo_rready_o),

        .fifo_wsel_o              (fifo_wsel_o),
        .fifo_wvalid_o            (fifo_wvalid_o),
        .fifo_wdata_sel_o         (fifo_wdata_sel_o),
        .fifo_wready_i            (fifo_wready_i),

        .message_length_i         (message_length_i),
        .sha_message_length_o     (sha_message_length),

        .idle_o                   (hmac_idle_o)
    );


    // ========================================================
    // OpenTitan SHA-2 engine
    // ========================================================

    prim_sha2_32 #(
        .MultimodeEn(1'b1)
    ) u_prim_sha2_32 (
        .clk_i                    (clk_i),
        .rst_ni                   (rst_ni),

        .wipe_secret_i            (1'b0),
        .wipe_v_i                 ('0),

        .fifo_rvalid_i            (shaf_rvalid),
        .fifo_rdata_i             (shaf_rdata),
        .fifo_rready_o            (shaf_rready),

        .sha_en_i                 (1'b1),

        .hash_start_i             (sha_hash_start),
        .hash_stop_i              (reg_hash_stop_i),
        .hash_continue_i          (sha_hash_continue),

        .digest_mode_i            (digest_size),

        .hash_process_i           (sha_hash_process),
        .hash_done_o              (sha_hash_done),

        .message_length_i         (sha_message_length),

        .digest_i                 (digest_initial),
        .digest_we_i              (digest_we),
        .digest_o                 (digest),

        .digest_on_blk_o          (digest_on_blk_o),
        .hash_running_o           (hash_running_o),
        .idle_o                   (sha_idle_o)
    );


    // ========================================================
    // SHA-256 digest
    //
    // SHA2-256 uses the lower 32 bits of each extended digest
    // word in the OpenTitan multimode SHA implementation.
    // ========================================================

    assign digest_o = {
        digest[0][31:0],
        digest[1][31:0],
        digest[2][31:0],
        digest[3][31:0],
        digest[4][31:0],
        digest[5][31:0],
        digest[6][31:0],
        digest[7][31:0]
    };

endmodule
