`timescale 1ns/1ps

module tb_opentitan_aes128_wrapper;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic start_i = 1'b0;
  logic [127:0] plaintext_i;
  logic [127:0] key_i;

  logic ready_o;
  logic busy_o;
  logic done_o;
  logic [127:0] ciphertext_o;
  logic alert_o;

  opentitan_aes128_wrapper dut (
    .clk_i,
    .rst_ni,
    .start_i,
    .plaintext_i,
    .key_i,
    .ready_o,
    .busy_o,
    .done_o,
    .ciphertext_o,
    .alert_o
  );

  always #5 clk_i = ~clk_i;

  task automatic run_kat(
    input logic [127:0] pt,
    input logic [127:0] key,
    input logic [127:0] expected
  );
    int cycles;
    begin
      while (!ready_o) @(posedge clk_i);

      plaintext_i = pt;
      key_i       = key;
      start_i     = 1'b1;
      @(posedge clk_i);
      start_i     = 1'b0;

      cycles = 0;
      while (!done_o && cycles < 100) begin
        @(posedge clk_i);
        cycles++;
      end

      if (!done_o) begin
        $display("FAIL: timeout waiting for done_o");
        $fatal(1);
      end

      $display("Plaintext : %032h", pt);
      $display("Key       : %032h", key);
      $display("Expected  : %032h", expected);
      $display("Observed  : %032h", ciphertext_o);
      $display("Cycles    : %0d", cycles);

      if (alert_o) begin
        $display("FAIL: AES core asserted alert_o");
        $fatal(1);
      end

      if (ciphertext_o !== expected) begin
        $display("FAIL: AES-128 known-answer mismatch");
        $fatal(1);
      end

      $display("PASS: AES-128 NIST known-answer test");
    end
  endtask

  initial begin
    plaintext_i = '0;
    key_i       = '0;
    start_i     = 1'b0;

    repeat (5) @(posedge clk_i);
    rst_ni = 1'b1;
    repeat (3) @(posedge clk_i);

    // FIPS-197 Appendix C AES-128 example.
    run_kat(
      128'h00112233445566778899aabbccddeeff,
      128'h000102030405060708090a0b0c0d0e0f,
      128'h69c4e0d86a7b0430d8cdb78070b4c55a
    );

    repeat (3) @(posedge clk_i);
    $display("SUCCESS: deterministic OpenTitan AES-128 wrapper is validated.");
    $finish;
  end

endmodule
