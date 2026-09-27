`timescale 1ns/1ps

module tb_opentitan_aes128_diag_batch;

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

  logic [8471:0] diag_o;

  integer fin;
  integer fout;
  integer ftrace;

  integer rc;
  integer idx;
  integer cycles;

  logic [127:0] pt;
  logic [127:0] key;

  string input_path;
  string output_path;
  string trace_path;

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
    .alert_o,
    .diag_o
  );

  always #5 clk_i = ~clk_i;


  task automatic run_one(
    input integer txn_idx,
    input logic [127:0] txn_pt,
    input logic [127:0] txn_key
  );
    begin

      // -------------------------------------------------------
      // Start only from a clean falling-edge boundary.
      // This guarantees stimulus is stable before the DUT's
      // active rising edge.
      // -------------------------------------------------------

      @(negedge clk_i);

      while (!(ready_o && !busy_o && !done_o))
        @(negedge clk_i);

      plaintext_i = txn_pt;
      key_i       = txn_key;
      start_i     = 1'b1;

      cycles = 0;

      // -------------------------------------------------------
      // Acceptance edge.
      // -------------------------------------------------------

      @(posedge clk_i);
      #1;

      if (!busy_o || done_o) begin
        $display(
          "FAIL: transaction %0d was not accepted correctly (ready=%b busy=%b done=%b)",
          txn_idx,
          ready_o,
          busy_o,
          done_o
        );
        $fatal(1);
      end

      // This is execution state #1.
      cycles = cycles + 1;

      $fdisplay(
        ftrace,
        "%03d %02d %02118h %b %b %b %b %032h",
        txn_idx,
        cycles,
        diag_o,
        ready_o,
        busy_o,
        done_o,
        alert_o,
        ciphertext_o
      );

      // Deassert safely between active clock edges.
      // Do NOT wait on another clock event here.
      #1;
      start_i = 1'b0;

      // -------------------------------------------------------
      // Capture only actual AES execution states.
      //
      // busy_o == 1  -> execution cycle
      // done_o == 1  -> terminal completion state, not a cycle
      // -------------------------------------------------------

      while (!done_o && cycles < 100) begin

        @(posedge clk_i);
        #1;

        if (busy_o) begin

          cycles = cycles + 1;

          $fdisplay(
            ftrace,
            "%03d %02d %02118h %b %b %b %b %032h",
            txn_idx,
            cycles,
            diag_o,
            ready_o,
            busy_o,
            done_o,
            alert_o,
            ciphertext_o
          );

        end

      end

      if (!done_o) begin
        $display(
          "FAIL: timeout at transaction %0d",
          txn_idx
        );
        $fatal(1);
      end

      if (alert_o) begin
        $display(
          "FAIL: alert_o asserted at transaction %0d",
          txn_idx
        );
        $fatal(1);
      end

      if (cycles != 12) begin
        $display(
          "FAIL: transaction %0d execution states=%0d expected=12",
          txn_idx,
          cycles
        );
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

    if (!$value$plusargs("TRACE=%s", trace_path)) begin
      $display("FAIL: +TRACE=<path> required");
      $fatal(1);
    end

    fin = $fopen(input_path, "r");

    if (fin == 0) begin
      $display(
        "FAIL: cannot open input file %s",
        input_path
      );
      $fatal(1);
    end

    fout = $fopen(output_path, "w");

    if (fout == 0) begin
      $display(
        "FAIL: cannot open output file %s",
        output_path
      );
      $fatal(1);
    end

    ftrace = $fopen(trace_path, "w");

    if (ftrace == 0) begin
      $display(
        "FAIL: cannot open trace file %s",
        trace_path
      );
      $fatal(1);
    end

    // Same reset sequence as validated batch TB.
    repeat (5)
      @(posedge clk_i);

    rst_ni = 1'b1;

    repeat (3)
      @(posedge clk_i);

    while (!$feof(fin)) begin

      rc = $fscanf(
        fin,
        "%d %h %h\n",
        idx,
        pt,
        key
      );

      if (rc == 3)
        run_one(idx, pt, key);

    end

    $fclose(fin);
    $fclose(fout);
    $fclose(ftrace);

    $display(
      "SUCCESS: completed AES diagnostic batch simulation."
    );

    $finish;

  end

endmodule
