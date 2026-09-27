`timescale 1ns/1ps

module tb_runtime_fault_batch;

  localparam integer N_CAND = 8472;

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

  integer fin, fout, rc, idx;
  integer ready_wait, done_wait;
  integer txn_count;

  integer fault_enable;
  integer fault_site;
  integer fault_stuck;

  reg [127:0] pt, key;
  string input_path, output_path;

  localparam integer READY_TIMEOUT_CYCLES = 32;
  localparam integer DONE_TIMEOUT_CYCLES  = 100;

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
    input integer txn_idx,
    input logic [127:0] txn_pt,
    input logic [127:0] txn_key
  );
    begin

      // Match the validated hard-coded fault simulator:
      // every transaction starts from an independent reset.
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
          "%03d READY_TIMEOUT %032h %032h %032h %0d %0d",
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
            "%03d DONE_TIMEOUT %032h %032h %032h %0d %0d",
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
            "%03d OK %032h %032h %032h %0d %0d",
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

    if (
      !$value$plusargs(
        "INPUT=%s",
        input_path
      )
    )
      $fatal(1, "Missing +INPUT");

    if (
      !$value$plusargs(
        "OUTPUT=%s",
        output_path
      )
    )
      $fatal(1, "Missing +OUTPUT");

    // Default = completely golden circuit.
    fault_enable = 0;
    fault_site = 0;
    fault_stuck = 0;

    void'(
      $value$plusargs(
        "FAULT_ENABLE=%d",
        fault_enable
      )
    );

    void'(
      $value$plusargs(
        "SITE=%d",
        fault_site
      )
    );

    void'(
      $value$plusargs(
        "STUCK=%d",
        fault_stuck
      )
    );

    if (
      fault_enable != 0
      && fault_enable != 1
    )
      $fatal(
        1,
        "FAULT_ENABLE must be 0 or 1"
      );

    if (
      fault_stuck != 0
      && fault_stuck != 1
    )
      $fatal(
        1,
        "STUCK must be 0 or 1"
      );

    fault_mask_i = '0;

    if (fault_enable == 1) begin

      if (
        fault_site < 0
        || fault_site >= N_CAND
      )
        $fatal(
          1,
          "SITE out of range: %0d",
          fault_site
        );

      fault_mask_i[
        fault_site
      ] = 1'b1;

    end

    fault_stuck_value_i = (
      fault_stuck
      ? 1'b1
      : 1'b0
    );

    if (fault_enable == 0) begin

      if (fault_mask_i !== '0)
        $fatal(
          1,
          "Golden mode has nonzero fault mask"
        );

      $display(
        "RUNTIME_FAULT_MODE: GOLDEN"
      );

    end else begin

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
        "RUNTIME_FAULT_MODE: SITE=%0d STUCK=%0d",
        fault_site,
        fault_stuck
      );

    end

    fin = $fopen(
      input_path,
      "r"
    );

    fout = $fopen(
      output_path,
      "w"
    );

    if (
      fin == 0
      || fout == 0
    )
      $fatal(
        1,
        "Cannot open I/O file"
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

        run_one(
          idx,
          pt,
          key
        );

        txn_count++;

      end
    end

    $fclose(fin);
    $fclose(fout);

    if (txn_count != 256)
      $fatal(
        1,
        "Expected 256 transactions, got %0d",
        txn_count
      );

    $display(
      "SUCCESS: runtime-fault batch complete: %0d transactions.",
      txn_count
    );

    $finish;
  end

endmodule
