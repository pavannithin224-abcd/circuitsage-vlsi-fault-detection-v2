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

  // -----------------------------------------------------------
  // TEMPORARY Stage 9B timing instrumentation.
  // PRE = active-region values at posedge.
  // POST = values after sequential/NBA updates settle.
  // -----------------------------------------------------------

  always @(posedge clk_i) begin
    $display(
      "DBG_POS_PRE  t=%0t idx=%0d start=%b ready=%b busy=%b done=%b",
      $time, idx, start_i, ready_o, busy_o, done_o
    );

    #1;

    $display(
      "DBG_POS_POST t=%0t idx=%0d start=%b ready=%b busy=%b done=%b",
      $time, idx, start_i, ready_o, busy_o, done_o
    );
  end

  always @(negedge clk_i) begin
    #1;

    $display(
      "DBG_NEG_POST t=%0t idx=%0d start=%b ready=%b busy=%b done=%b",
      $time, idx, start_i, ready_o, busy_o, done_o
    );
  end


  task automatic run_one(
    input integer txn_idx,
    input logic [127:0] txn_pt,
    input logic [127:0] txn_key
  );
    begin

      // -------------------------------------------------------
      // Race-free handshake:
      //   - wait for ready_o at a falling edge
      //   - drive inputs/start_i at that falling edge
      //   - DUT samples them at the next rising edge
      // -------------------------------------------------------

      // Wait for a truly idle transaction boundary.
      //
      // ready_o alone is insufficient because the wrapper may assert
      // ready_o during the final done_o pulse of the previous AES job.
      //
      // Starting while done_o is still high causes an inter-transaction
      // one-cycle phase shift.
      while (!(ready_o && !busy_o && !done_o))
        @(negedge clk_i);

      // Drive stimulus on a falling edge so it is stable well before
      // the next active rising edge.
      plaintext_i = txn_pt;
      key_i       = txn_key;
      start_i     = 1'b1;

      // Acceptance edge.
      @(posedge clk_i);
      #1;

      cycles = 1;

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

      // Deassert start_i only on a falling edge.
      @(negedge clk_i);
      start_i = 1'b0;

      // Capture subsequent settled execution states.
      while (!done_o && cycles < 100) begin

        @(posedge clk_i);
        #1;

        cycles++;

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

      $display(
        "DBG_LATENCY txn=%0d settled_cycles=%0d",
        txn_idx,
        cycles
      );

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
