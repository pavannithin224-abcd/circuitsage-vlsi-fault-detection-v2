`timescale 1ns/1ps

module tb_runtime_fault_temporal_cipher_smoke;

  localparam integer N_CAND = 8472;
  localparam integer N_VEC  = 256;

  localparam integer READY_TIMEOUT_CYCLES = 32;
  localparam integer DONE_TIMEOUT_CYCLES  = 100;

  logic clk_i = 1'b0;
  logic rst_ni = 1'b0;
  logic start_i = 1'b0;

  logic [127:0] plaintext_i = '0;
  logic [127:0] key_i = '0;

  logic [N_CAND-1:0] fault_mask_i = '0;
  logic fault_stuck_value_i = 1'b0;

  wire ready_o;
  wire busy_o;
  wire done_o;
  wire [127:0] ciphertext_o;
  wire alert_o;

  integer fin;
  integer fplan;
  integer ftrace;
  integer fsummary;

  integer rc;
  integer idx;

  integer ready_wait;
  integer done_wait;

  integer t;
  integer txn_count;
  integer fault_count;

  integer fault_id;
  integer fault_site;
  integer fault_stuck;

  integer max_tx;

  reg [127:0] pt;
  reg [127:0] key;

  integer tx_idx_mem [0:N_VEC-1];
  reg [127:0] pt_mem [0:N_VEC-1];
  reg [127:0] key_mem [0:N_VEC-1];

  string input_path;
  string fault_plan_path;
  string trace_path;
  string summary_path;


  opentitan_aes128_wrapper dut (
    .clk_i(clk_i),
    .rst_ni(rst_ni),
    .start_i(start_i),
    .plaintext_i(plaintext_i),
    .key_i(key_i),

    .fault_mask_i(fault_mask_i),
    .fault_stuck_value_i(fault_stuck_value_i),

    .ready_o(ready_o),
    .busy_o(busy_o),
    .done_o(done_o),
    .ciphertext_o(ciphertext_o),
    .alert_o(alert_o)
  );


  always #5 clk_i = ~clk_i;


  task automatic reset_dut;
    begin

      start_i = 1'b0;
      plaintext_i = '0;
      key_i = '0;

      rst_ni = 1'b0;
      repeat (5) @(posedge clk_i);

      rst_ni = 1'b1;
      repeat (3) @(posedge clk_i);

    end
  endtask


  task automatic run_one(
    input integer fid,
    input integer site,
    input integer stuck,
    input integer txn_idx,
    input logic [127:0] txn_pt,
    input logic [127:0] txn_key
  );

    begin

      reset_dut();

      ready_wait = 0;

      while (
        (ready_o !== 1'b1)
        && ready_wait < READY_TIMEOUT_CYCLES
      ) begin

        @(posedge clk_i);
        #1;

        ready_wait++;

        $fdisplay(
          ftrace,
          "%0d %0d %0d %03d READY %0d %0d %0d %0d %0d %032h",
          fid,
          site,
          stuck,
          txn_idx,
          ready_wait,
          (ready_o === 1'b1),
          (busy_o === 1'b1),
          (done_o === 1'b1),
          (alert_o === 1'b1),
          ciphertext_o
        );

      end


      if (ready_o !== 1'b1) begin

        $fdisplay(
          fsummary,
          "%0d %0d %0d %03d READY_TIMEOUT %032h %032h %032h %0d %0d",
          fid,
          site,
          stuck,
          txn_idx,
          txn_pt,
          txn_key,
          128'b0,
          ready_wait,
          (alert_o === 1'b1)
        );

      end else begin

        plaintext_i = txn_pt;
        key_i = txn_key;

        start_i = 1'b1;

        @(posedge clk_i);

        start_i = 1'b0;

        #1;

        // Cycle 0 = settled external state immediately
        // after the start transaction has been accepted.
        $fdisplay(
          ftrace,
          "%0d %0d %0d %03d EXEC %0d %0d %0d %0d %0d %032h",
          fid,
          site,
          stuck,
          txn_idx,
          0,
          (ready_o === 1'b1),
          (busy_o === 1'b1),
          (done_o === 1'b1),
          (alert_o === 1'b1),
          ciphertext_o
        );

        done_wait = 0;

        while (
          (done_o !== 1'b1)
          && done_wait < DONE_TIMEOUT_CYCLES
        ) begin

          @(posedge clk_i);
          #1;

          done_wait++;

          $fdisplay(
            ftrace,
            "%0d %0d %0d %03d EXEC %0d %0d %0d %0d %0d %032h",
            fid,
            site,
            stuck,
            txn_idx,
            done_wait,
            (ready_o === 1'b1),
            (busy_o === 1'b1),
            (done_o === 1'b1),
            (alert_o === 1'b1),
          ciphertext_o
          );

        end


        if (done_o !== 1'b1) begin

          $fdisplay(
            fsummary,
            "%0d %0d %0d %03d DONE_TIMEOUT %032h %032h %032h %0d %0d",
            fid,
            site,
            stuck,
            txn_idx,
            txn_pt,
            txn_key,
            128'b0,
            done_wait,
            (alert_o === 1'b1)
          );

        end else begin

          $fdisplay(
            fsummary,
            "%0d %0d %0d %03d OK %032h %032h %032h %0d %0d",
            fid,
            site,
            stuck,
            txn_idx,
            txn_pt,
            txn_key,
            ciphertext_o,
            (done_wait + 1),
            (alert_o === 1'b1)
          );

        end

      end

    end
  endtask


  initial begin

    if (!$value$plusargs("INPUT=%s", input_path))
      $fatal(1, "Missing +INPUT");

    if (!$value$plusargs("FAULT_PLAN=%s", fault_plan_path))
      $fatal(1, "Missing +FAULT_PLAN");

    if (!$value$plusargs("TRACE_OUTPUT=%s", trace_path))
      $fatal(1, "Missing +TRACE_OUTPUT");

    if (!$value$plusargs("SUMMARY_OUTPUT=%s", summary_path))
      $fatal(1, "Missing +SUMMARY_OUTPUT");

    max_tx = 8;

    if (!$value$plusargs("MAX_TX=%d", max_tx))
      max_tx = 8;

    if (
      max_tx < 1
      || max_tx > N_VEC
    )
      $fatal(
        1,
        "MAX_TX must be in range 1..256"
      );


    // ------------------------------------------------------
    // Load complete 256-vector bank.
    // ------------------------------------------------------

    fin = $fopen(
      input_path,
      "r"
    );

    if (fin == 0)
      $fatal(
        1,
        "Cannot open INPUT"
      );

    txn_count = 0;

    while (!$feof(fin)) begin

      rc = $fscanf(
        fin,
        "%d %h %h\n",
        idx,
        pt,
        key
      );

      if (rc == 3) begin

        if (txn_count >= N_VEC)
          $fatal(
            1,
            "More than 256 vectors"
          );

        if (idx != txn_count)
          $fatal(
            1,
            "Expected vector %0d, got %0d",
            txn_count,
            idx
          );

        tx_idx_mem[txn_count] = idx;
        pt_mem[txn_count] = pt;
        key_mem[txn_count] = key;

        txn_count++;

      end

    end

    $fclose(fin);

    if (txn_count != N_VEC)
      $fatal(
        1,
        "Expected 256 vectors, got %0d",
        txn_count
      );


    fplan = $fopen(
      fault_plan_path,
      "r"
    );

    ftrace = $fopen(
      trace_path,
      "w"
    );

    fsummary = $fopen(
      summary_path,
      "w"
    );

    if (fplan == 0)
      $fatal(1, "Cannot open FAULT_PLAN");

    if (ftrace == 0)
      $fatal(1, "Cannot open TRACE_OUTPUT");

    if (fsummary == 0)
      $fatal(1, "Cannot open SUMMARY_OUTPUT");


    fault_count = 0;

    while (!$feof(fplan)) begin

      rc = $fscanf(
        fplan,
        "%d %d %d\n",
        fault_id,
        fault_site,
        fault_stuck
      );

      if (rc == 3) begin

        if (
          fault_site < 0
          || fault_site >= N_CAND
        )
          $fatal(
            1,
            "SITE out of range"
          );

        if (
          fault_stuck != 0
          && fault_stuck != 1
        )
          $fatal(
            1,
            "STUCK must be 0/1"
          );

        fault_mask_i = '0;
        fault_mask_i[fault_site] = 1'b1;

        fault_stuck_value_i = (
          fault_stuck != 0
        );

        if ($countones(fault_mask_i) != 1)
          $fatal(
            1,
            "Fault mask not one-hot"
          );

        for (
          t = 0;
          t < max_tx;
          t = t + 1
        ) begin

          run_one(
            fault_id,
            fault_site,
            fault_stuck,
            tx_idx_mem[t],
            pt_mem[t],
            key_mem[t]
          );

        end

        fault_count++;

      end

    end

    $fclose(fplan);
    $fclose(ftrace);
    $fclose(fsummary);

    if (fault_count == 0)
      $fatal(
        1,
        "FAULT_PLAN contained zero faults"
      );

    fault_mask_i = '0;
    fault_stuck_value_i = 1'b0;

    $display(
      "SUCCESS: temporal runtime batch complete: %0d faults x %0d vectors.",
      fault_count,
      max_tx
    );

    $finish;

  end

endmodule
