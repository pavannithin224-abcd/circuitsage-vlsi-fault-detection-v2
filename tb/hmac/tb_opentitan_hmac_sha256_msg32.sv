`timescale 1ns/1ps

module tb_opentitan_hmac_sha256_msg32;

    logic clk_i;
    logic rst_ni;

    logic start_i;

    logic [255:0] key_i;
    logic [255:0] message_i;

    logic busy_o;
    logic done_o;
    logic [255:0] digest_o;

    localparam logic [255:0] TEST_KEY =
        256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;

    localparam logic [255:0] TEST_MESSAGE =
        256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;

    localparam logic [255:0] EXPECTED =
        256'he8499be4f1980d68f13222a418df5cbd97d53fddf590c2108e22d40005b70713;


    opentitan_hmac_sha256_msg32 dut (
        .clk_i      (clk_i),
        .rst_ni     (rst_ni),

        .start_i    (start_i),

        .key_i      (key_i),
        .message_i  (message_i),

        .busy_o     (busy_o),
        .done_o     (done_o),
        .digest_o   (digest_o)
    );


    initial begin
        clk_i = 1'b0;
        forever #5 clk_i = ~clk_i;
    end


    initial begin : test_sequence

        int cycles;

        rst_ni    = 1'b0;
        start_i   = 1'b0;

        key_i     = TEST_KEY;
        message_i = TEST_MESSAGE;

        repeat (5) @(posedge clk_i);

        #1;
        rst_ni = 1'b1;

        repeat (3) @(posedge clk_i);

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

            $display(
                "HARDWARE RESULT : TIMEOUT"
            );

            $fatal(
                1,
                "HMAC hardware did not complete"
            );

        end


        $display(
            "========================================"
        );

        $display(
            "OPENTITAN HMAC-SHA256 FUNCTIONAL TEST"
        );

        $display(
            "========================================"
        );

        $display(
            "Cycles   : %0d",
            cycles
        );

        $display(
            "Key      : %064h",
            TEST_KEY
        );

        $display(
            "Message  : %064h",
            TEST_MESSAGE
        );

        $display(
            "Expected : %064h",
            EXPECTED
        );

        $display(
            "Actual   : %064h",
            digest_o
        );


        if (digest_o === EXPECTED) begin

            $display(
                "RESULT   : PASS"
            );

            $display(
                "OPENTITAN HMAC-SHA256 WRAPPER WORKING"
            );

            $display(
                "========================================"
            );

            $finish;

        end else begin

            $display(
                "RESULT   : FAIL"
            );

            $display(
                "========================================"
            );

            $fatal(
                1,
                "HMAC digest mismatch"
            );

        end

    end

endmodule
