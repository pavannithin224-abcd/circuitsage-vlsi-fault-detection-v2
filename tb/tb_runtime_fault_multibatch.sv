`timescale 1ns/1ps

module tb_runtime_fault_multibatch;

  localparam integer N_CAND   = 8472;
  localparam integer TRACE_LEN = 256;

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
  integer fout;

  integer rc;
  integer idx;

  integer ready_wait;
  integer done_wait;

  integer txn_count;
  integer fault_count;
  integer t;

  integer fault_id;
  integer fault_site;
  integer fault_stuck;

  reg [127:0] pt;
  reg [127:0] key;

  integer tx_idx_mem [0:TRACE_LEN-1];
  reg [127:0] pt_mem [0:TRACE_LEN-1];
  reg [127:0] key_mem [0:TRACE_LEN-1];

  string input_path;
  string fault_plan_path;
  string output_path;


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

      // Preserve validated single-fault semantics:
      // independent reset before every transaction.
      reset_dut();

      ready_wait = 0;

      while (
        (ready_o !== 1'b1)
        && ready_wait < READY_TIMEOUT_CYCLES
      ) begin

        @(posedge clk_i);
        ready_wait++;

      end


      if (ready_o !== 1'b1) begin

        $fdisplay(
          fout,
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

        done_wait = 0;

        while (
          (done_o !== 1'b1)
          && done_wait < DONE_TIMEOUT_CYCLES
        ) begin

          @(posedge clk_i);
          done_wait++;

        end


        if (done_o !== 1'b1) begin

          $fdisplay(
            fout,
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
            fout,
            "%0d %0d %0d %03d OK %032h %032h %032h %0d %0d",
            fid,
            site,
            stuck,
            txn_idx,
            txn_pt,
            txn_key,
            ciphertext_o,
            done_wait,
            (alert_o === 1'b1)
          );

        end

      end

    end

  endtask


  initial begin

    // ------------------------------------------------------
    // Required arguments
    // ------------------------------------------------------

    if (
      !$value$plusargs(
        "INPUT=%s",
        input_path
      )
    )
      $fatal(
        1,
        "Missing +INPUT"
      );


    if (
      !$value$plusargs(
        "FAULT_PLAN=%s",
        fault_plan_path
      )
    )
      $fatal(
        1,
        "Missing +FAULT_PLAN"
      );


    if (
      !$value$plusargs(
        "OUTPUT=%s",
        output_path
      )
    )
      $fatal(
        1,
        "Missing +OUTPUT"
      );


    // ------------------------------------------------------
    // Load vector bank ONCE.
    // ------------------------------------------------------

    fin = $fopen(
      input_path,
      "r"
    );

    if (fin == 0)
      $fatal(
        1,
        "Cannot open INPUT file"
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

        if (
          txn_count >= TRACE_LEN
        )
          $fatal(
            1,
            "INPUT has more than 256 transactions"
          );

        if (
          idx != txn_count
        )
          $fatal(
            1,
            "Expected transaction index %0d, got %0d",
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


    if (
      txn_count != TRACE_LEN
    )
      $fatal(
        1,
        "Expected 256 transactions, got %0d",
        txn_count
      );


    // ------------------------------------------------------
    // Open fault plan and output.
    //
    // Plan format:
    //
    //   fault_id candidate_position stuck_value
    //
    // Example:
    //   0 290 0
    //   1 333 1
    // ------------------------------------------------------

    fplan = $fopen(
      fault_plan_path,
      "r"
    );

    fout = $fopen(
      output_path,
      "w"
    );


    if (fplan == 0)
      $fatal(
        1,
        "Cannot open FAULT_PLAN file"
      );

    if (fout == 0)
      $fatal(
        1,
        "Cannot open OUTPUT file"
      );


    fault_count = 0;

    // ------------------------------------------------------
    // Process every fault hypothesis in one simulator run.
    // ------------------------------------------------------

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
            "SITE out of range: %0d",
            fault_site
          );


        if (
          fault_stuck != 0
          && fault_stuck != 1
        )
          $fatal(
            1,
            "STUCK must be 0 or 1"
          );


        // Select exactly one runtime fault.
        fault_mask_i = '0;

        fault_mask_i[
          fault_site
        ] = 1'b1;

        fault_stuck_value_i = (
          fault_stuck != 0
        );


        if (
          $countones(
            fault_mask_i
          ) != 1
        )
          $fatal(
            1,
            "Runtime fault mask is not one-hot"
          );


        $display(
          "MULTIBATCH_FAULT_BEGIN: ID=%0d SITE=%0d STUCK=%0d",
          fault_id,
          fault_site,
          fault_stuck
        );


        for (
          t = 0;
          t < TRACE_LEN;
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


        $display(
          "MULTIBATCH_FAULT_END: ID=%0d SITE=%0d STUCK=%0d TRANSACTIONS=%0d",
          fault_id,
          fault_site,
          fault_stuck,
          TRACE_LEN
        );

      end

    end


    $fclose(fplan);
    $fclose(fout);


    if (
      fault_count == 0
    )
      $fatal(
        1,
        "FAULT_PLAN contained zero valid faults"
      );


    // Clear fault before ending simulation.
    fault_mask_i = '0;
    fault_stuck_value_i = 1'b0;


    $display(
      "SUCCESS: runtime-fault multibatch complete: %0d faults, %0d transactions.",
      fault_count,
      fault_count * TRACE_LEN
    );


    $finish;

  end

endmodule
