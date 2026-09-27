`timescale 1ns/1ps

module tb_opentitan_hmac_sha256_coverage_targeted;

    logic clk_i;
    logic rst_ni;
    logic start_i;

    logic [255:0] key_i;
    logic [255:0] message_i;

    logic busy_o;
    logic done_o;
    logic [255:0] digest_o;

    int pass_count;

    opentitan_hmac_sha256_msg32 dut (
        .clk_i     (clk_i),
        .rst_ni    (rst_ni),
        .start_i   (start_i),
        .key_i     (key_i),
        .message_i (message_i),
        .busy_o    (busy_o),
        .done_o    (done_o),
        .digest_o  (digest_o)
    );

    initial begin
        clk_i = 1'b0;
        forever #5 clk_i = ~clk_i;
    end

    task automatic apply_reset;
        begin
            start_i = 1'b0;
            rst_ni  = 1'b0;

            repeat (5) @(posedge clk_i);

            #1;
            rst_ni = 1'b1;

            repeat (3) @(posedge clk_i);
            #1;
        end
    endtask

    task automatic run_case(
        input int unsigned test_id,
        input logic [255:0] test_key,
        input logic [255:0] test_message,
        input logic [255:0] expected_digest
    );
        int cycles;

        begin
            while (busy_o) begin
                @(posedge clk_i);
                #1;
            end

            key_i     = test_key;
            message_i = test_message;
            start_i   = 1'b1;

            @(posedge clk_i);
            #1;
            start_i = 1'b0;

            cycles = 0;

            while (!done_o && cycles < 20000) begin
                @(posedge clk_i);
                #1;
                cycles++;
            end

            if (!done_o) begin
                $fatal(
                    1,
                    "TARGETED CASE %0d TIMEOUT",
                    test_id
                );
            end

            if (digest_o !== expected_digest) begin
                $display(
                    "Expected: %064h",
                    expected_digest
                );

                $display(
                    "Actual  : %064h",
                    digest_o
                );

                $fatal(
                    1,
                    "TARGETED CASE %0d DIGEST MISMATCH",
                    test_id
                );
            end

            pass_count++;

            $display(
                "TARGETED CASE %0d PASS cycles=%0d digest_bit6=%0d digest=%064h",
                test_id,
                cycles,
                digest_o[6],
                digest_o
            );

            @(posedge clk_i);
            #1;
        end
    endtask

    task automatic interrupt_with_reset(
        input logic [255:0] test_key,
        input logic [255:0] test_message
    );
        begin
            while (busy_o) begin
                @(posedge clk_i);
                #1;
            end

            key_i     = test_key;
            message_i = test_message;
            start_i   = 1'b1;

            @(posedge clk_i);
            #1;
            start_i = 1'b0;

            repeat (24) begin
                @(posedge clk_i);
                #1;
            end

            if (!busy_o) begin
                $fatal(
                    1,
                    "Expected active transaction before reset"
                );
            end

            // Exercise the previously missed 1-to-0 reset toggle.
            rst_ni  = 1'b0;
            start_i = 1'b0;

            repeat (5) @(posedge clk_i);
            #1;

            if (busy_o !== 1'b0) begin
                $fatal(
                    1,
                    "busy_o did not clear during reset"
                );
            end

            if (done_o !== 1'b0) begin
                $fatal(
                    1,
                    "done_o did not clear during reset"
                );
            end

            rst_ni = 1'b1;

            repeat (3) @(posedge clk_i);
            #1;

            $display(
                "MID_TRANSACTION_RESET=PASS"
            );
        end
    endtask

    initial begin : targeted_sequence

        pass_count = 0;
        rst_ni     = 1'b0;
        start_i    = 1'b0;
        key_i      = '0;
        message_i  = '0;

        apply_reset();

        // Forces digest_o[6] from its reset value 0 to 1.
        run_case(
            0,
            256'h78c57b4697527984ad0ce6b516fd22a16f38a528b1cedbdd4adeb2948ee537a5,
            256'h200b4e58e17e55dfb3967df6499038871a29d98f4041ed4a31197223b7a7afcb,
            256'he2b8a80fb25c04223911b8c5a51366b0c43ae6f1fed97e4513c5cd2ed53ed94e
        );

        // Forces digest_o[6] from 1 back to 0.
        run_case(
            1,
            256'hb02a75de17edcbb8e5fbd860a0f4b425abd76c80672a1241544bd8fb258f66de,
            256'hb979ddb3afe8ee658dee9f333b2641e030812d544c906c77267e8b679afa6e4f,
            256'h318c8c1d1d1dd32b17c24d64a361ca4d4789ad35d24f2d9d092c03f582e00b8c
        );

        // Abort an active transaction with reset.
        interrupt_with_reset(
            256'h7716626d4ea02a4e15a3522b83a80fd7e2322efc0ee1a3a5f2c17fb1d61f9cea,
            256'h9ba691acd3f0c8561d77404d3aa034e9ce9affc3aecec51c5a25ed82e0ffc60e
        );

        // Verify complete functional recovery after interruption.
        run_case(
            2,
            256'h7716626d4ea02a4e15a3522b83a80fd7e2322efc0ee1a3a5f2c17fb1d61f9cea,
            256'h9ba691acd3f0c8561d77404d3aa034e9ce9affc3aecec51c5a25ed82e0ffc60e,
            256'hf9a90a18e1aec88b322113c118e958778bc7e09b6f9082af232863316682d0eb
        );

        $display(
            "========================================"
        );

        $display(
            "HMAC TARGETED COVERAGE REGRESSION"
        );

        $display(
            "TARGETED_CASES_PASSED=%0d",
            pass_count
        );

        $display(
            "MID_TRANSACTION_RESET=PASS"
        );

        if (pass_count == 3) begin
            $display(
                "TARGETED_COVERAGE_RESULT=PASS"
            );

            $finish;
        end else begin
            $display(
                "TARGETED_COVERAGE_RESULT=FAIL"
            );

            $fatal(
                1,
                "Targeted coverage regression failed"
            );
        end
    end

endmodule
