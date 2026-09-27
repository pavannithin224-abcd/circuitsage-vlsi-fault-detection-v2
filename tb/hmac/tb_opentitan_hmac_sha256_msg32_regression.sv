`timescale 1ns/1ps

module tb_opentitan_hmac_sha256_msg32_regression;

    logic clk_i;
    logic rst_ni;
    logic start_i;

    logic [255:0] key_i;
    logic [255:0] message_i;

    logic busy_o;
    logic done_o;
    logic [255:0] digest_o;

    int pass_count;
    int fail_count;

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

            #1;
            start_i = 1'b1;

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
                    "TEST %0d TIMEOUT",
                    test_id
                );
            end

            if (digest_o === expected_digest) begin
                pass_count++;

                $display(
                    "TEST %0d PASS cycles=%0d digest=%064h",
                    test_id,
                    cycles,
                    digest_o
                );
            end else begin
                fail_count++;

                $display(
                    "TEST %0d FAIL cycles=%0d",
                    test_id,
                    cycles
                );

                $display(
                    "  Expected: %064h",
                    expected_digest
                );

                $display(
                    "  Actual  : %064h",
                    digest_o
                );
            end

            // Allow done_o to clear before the next transaction.
            @(posedge clk_i);
            #1;
        end
    endtask

    initial begin : regression_sequence

        rst_ni    = 1'b0;
        start_i   = 1'b0;
        key_i     = '0;
        message_i = '0;

        pass_count = 0;
        fail_count = 0;

        repeat (5) @(posedge clk_i);

        #1;
        rst_ni = 1'b1;

        repeat (3) @(posedge clk_i);

        run_case(
            0,
            256'h0000000000000000000000000000000000000000000000000000000000000000,
            256'h0000000000000000000000000000000000000000000000000000000000000000,
            256'h33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a
        );

        run_case(
            1,
            256'hffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
            256'hffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff,
            256'h8a5183c87dc4e694cd5d35870edc9b2fbe05c71dbf9c5b17648a25f816fe292d
        );

        run_case(
            2,
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f,
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f,
            256'he8499be4f1980d68f13222a418df5cbd97d53fddf590c2108e22d40005b70713
        );

        run_case(
            3,
            256'h1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100,
            256'h1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100,
            256'hb648bddbdcf9c5bbe9e2d8180d3230fb5eb444c9e023367e1ab829cb2f260589
        );

        run_case(
            4,
            256'haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
            256'h5555555555555555555555555555555555555555555555555555555555555555,
            256'h9fa02bd3e284043e573b4493eb2312a3511c0ca5dfa888f9ecd59d34bc7d7222
        );

        run_case(
            5,
            256'h5555555555555555555555555555555555555555555555555555555555555555,
            256'haaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,
            256'h154270ac462b338d703498160852178610161fa6de569bfe21b7e30738fd110a
        );

        run_case(
            6,
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f,
            256'h1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100,
            256'h9d2a0fddbe2de00ed0a9d9ec544d6be4be7b82ae931ce098c4ddfd326afeb11c
        );

        run_case(
            7,
            256'h1f1e1d1c1b1a191817161514131211100f0e0d0c0b0a09080706050403020100,
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f,
            256'h357b84d95089aeb35cd62ac28890748a4f92a9d8970129777f3e8bdf91ed8d98
        );

        run_case(
            8,
            256'h0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef,
            256'hfedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210,
            256'hfc620ba9fee2a44f2ea7a4cdf04348f2fa7299feb84ea028c48f80bba0bdddb0
        );

        run_case(
            9,
            256'h6a09e667bb67ae853c6ef372a54ff53a510e527f9b05688c1f83d9ab5be0cd19,
            256'h243f6a8885a308d313198a2e03707344a4093822299f31d0082efa98ec4e6c89,
            256'h1836585f1552b7f3cdbf7cb4dd2ad4c3d0e55e338cf648a39caab567cc192c01
        );

        $display("========================================");
        $display("HMAC-SHA256 MULTI-VECTOR REGRESSION");
        $display("TOTAL TESTS : %0d", pass_count + fail_count);
        $display("PASSED      : %0d", pass_count);
        $display("FAILED      : %0d", fail_count);
        $display("========================================");

        if (fail_count == 0 && pass_count == 10) begin
            $display("REGRESSION_RESULT=PASS");
            $finish;
        end else begin
            $display("REGRESSION_RESULT=FAIL");
            $fatal(1, "HMAC regression failure");
        end
    end

endmodule
