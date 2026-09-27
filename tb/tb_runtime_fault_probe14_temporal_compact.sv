`timescale 1ns/1ps

module tb_runtime_fault_probe14_temporal_compact;

  logic [13:0] probe14_o;


  localparam integer N_CAND = 8472;
  localparam integer N_VEC  = 256;

  localparam integer READY_TIMEOUT_CYCLES = 32;
  localparam integer DONE_TIMEOUT_CYCLES  = 100;

  // EXEC cycles 0..100 inclusive = 101 cycles.
  // Each cycle stores:
  //
  //   {ready, busy, done, alert}
  //
  // 101 * 4 = 404 bits.
  localparam integer TEMP_CYCLES = 101;
  localparam integer TEMP_BITS   = 404;
  localparam integer PROBE14_BITS = 1414;

  logic [PROBE14_BITS-1:0] probe14_bits;


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
    .alert_o(alert_o),
    .probe14_o(probe14_o)
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

    integer ready_wait;
    integer done_wait;
    integer first_alert;

    logic [TEMP_BITS-1:0] temporal_bits;

    begin

      temporal_bits = '0;
      probe14_bits  = '0;
      first_alert = -1;

      reset_dut();

      ready_wait = 0;

      while (
        (ready_o !== 1'b1)
        && ready_wait < READY_TIMEOUT_CYCLES
      ) begin

        @(posedge clk_i);
        #1;

        ready_wait++;

      end


      if (ready_o !== 1'b1) begin

        $fdisplay(
          fout,
          "%0d %0d %0d %03d READY_TIMEOUT %h %0d %0d",
          fid,
          site,
          stuck,
          txn_idx,
          temporal_bits,
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

        // --------------------------------------------------
        // EXEC cycle 0.
        //
        // Bit layout for each nibble:
        //
        // bit 3 = ready
        // bit 2 = busy
        // bit 1 = done
        // bit 0 = alert
        //
        // temporal_bits[cycle*4 +: 4]
        // --------------------------------------------------

        temporal_bits[
          0 +: 4
        ] = {
          (ready_o === 1'b1),
          (busy_o === 1'b1),
          (done_o === 1'b1),
          (alert_o === 1'b1)
        };

        if (
          (alert_o === 1'b1)
          && first_alert < 0
        )
          first_alert = 0;

        done_wait = 0;


        while (
          (done_o !== 1'b1)
          && done_wait < DONE_TIMEOUT_CYCLES
        ) begin

          @(posedge clk_i);
          #1;

          done_wait++;

          temporal_bits[
            done_wait * 4 +: 4
          ] = {
            (ready_o === 1'b1),
            (busy_o === 1'b1),
            (done_o === 1'b1),
            (alert_o === 1'b1)
          };
          probe14_bits[done_wait * 14 +: 14] = probe14_o;

          if (
            (alert_o === 1'b1)
            && first_alert < 0
          )
            first_alert = done_wait;

        end


        if (done_o !== 1'b1) begin

          $fdisplay(
            fout,
            "%0d %0d %0d %03d DONE_TIMEOUT %h %0d %0d %0d %h",
            fid,
            site,
            stuck,
            txn_idx,
            temporal_bits,
            done_wait,
            (alert_o === 1'b1),
            first_alert,
            probe14_bits
          );

        end else begin

          // +1 preserves validated legacy cycle reporting.
          $fdisplay(
            fout,
            "%0d %0d %0d %03d OK %h %0d %0d %0d %h",
            fid,
            site,
            stuck,
            txn_idx,
            temporal_bits,
            (done_wait + 1),
            (alert_o === 1'b1),
            first_alert,
            probe14_bits
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

    if (!$value$plusargs("OUTPUT=%s", output_path))
      $fatal(1, "Missing +OUTPUT");

    max_tx = 8;

    if (!$value$plusargs("MAX_TX=%d", max_tx))
      max_tx = 8;

    if (
      max_tx < 1
      || max_tx > N_VEC
    )
      $fatal(
        1,
        "MAX_TX must be 1..256"
      );


    // ------------------------------------------------------
    // Load 256-vector bank once.
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

        tx_idx_mem[
          txn_count
        ] = idx;

        pt_mem[
          txn_count
        ] = pt;

        key_mem[
          txn_count
        ] = key;

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

    fout = $fopen(
      output_path,
      "w"
    );

    if (fplan == 0)
      $fatal(
        1,
        "Cannot open FAULT_PLAN"
      );

    if (fout == 0)
      $fatal(
        1,
        "Cannot open OUTPUT"
      );


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
    $fclose(fout);

    if (fault_count == 0)
      $fatal(
        1,
        "FAULT_PLAN contained zero faults"
      );

    fault_mask_i = '0;
    fault_stuck_value_i = 1'b0;

    $display(
      "SUCCESS: compact temporal batch complete: %0d faults x %0d vectors.",
      fault_count,
      max_tx
    );

    $finish;

  end

endmodule
