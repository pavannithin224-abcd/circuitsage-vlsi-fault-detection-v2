`timescale 1ns/1ps

module tb_opentitan_aes128_batch;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic start_i = 1'b0;
  logic [127:0] plaintext_i = '0;
  logic [127:0] key_i = '0;

  logic ready_o;
  logic busy_o;
  logic done_o;
  logic [127:0] ciphertext_o;
  logic alert_o;

  integer fin;
  integer fout;
  integer rc;
  integer idx;
  integer cycles;

  logic [127:0] pt;
  logic [127:0] key;

  string input_path;
  string output_path;

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

  task automatic run_one(
    input integer txn_idx,
    input logic [127:0] txn_pt,
    input logic [127:0] txn_key
  );
    begin
      while (!ready_o) @(posedge clk_i);

      plaintext_i = txn_pt;
      key_i       = txn_key;
      start_i     = 1'b1;
      @(posedge clk_i);
      start_i     = 1'b0;

      cycles = 0;
      while (!done_o && cycles < 100) begin
        @(posedge clk_i);
        cycles++;
      end

      if (!done_o) begin
        $display("FAIL: timeout at transaction %0d", txn_idx);
        $fatal(1);
      end

      if (alert_o) begin
        $display("FAIL: alert_o asserted at transaction %0d", txn_idx);
        $fatal(1);
      end

      $fdisplay(
        fout,
        "%03d %032h %032h %032h %0d",
        txn_idx,
        txn_pt,
        txn_key,
        ciphertext_o,
        cycles
      );
    end
  endtask

  initial begin
    if (!$value$plusargs("INPUT=%s", input_path)) begin
      $display("FAIL: +INPUT=<path> required");
      $fatal(1);
    end

    if (!$value$plusargs("OUTPUT=%s", output_path)) begin
      $display("FAIL: +OUTPUT=<path> required");
      $fatal(1);
    end

    fin = $fopen(input_path, "r");
    if (fin == 0) begin
      $display("FAIL: cannot open input file %s", input_path);
      $fatal(1);
    end

    fout = $fopen(output_path, "w");
    if (fout == 0) begin
      $display("FAIL: cannot open output file %s", output_path);
      $fatal(1);
    end

    repeat (5) @(posedge clk_i);
    rst_ni = 1'b1;
    repeat (3) @(posedge clk_i);

    while (!$feof(fin)) begin
      rc = $fscanf(fin, "%d %h %h\n", idx, pt, key);
      if (rc == 3) begin
        run_one(idx, pt, key);
      end
    end

    $fclose(fin);
    $fclose(fout);

    $display("SUCCESS: completed AES-128 batch simulation.");
    $finish;
  end

endmodule
