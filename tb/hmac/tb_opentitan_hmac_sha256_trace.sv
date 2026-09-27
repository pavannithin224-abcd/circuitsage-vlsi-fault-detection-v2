`timescale 1ns/1ps

module tb_opentitan_hmac_sha256_trace;

    logic clk_i;
    logic rst_ni;
    logic start_i;

    logic [255:0] key_i;
    logic [255:0] message_i;

    logic busy_o;
    logic done_o;
    logic [255:0] digest_o;

    integer sha_word_count;
    integer feedback_count;


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


    always @(posedge clk_i) begin

        if (!rst_ni) begin

            sha_word_count = 0;
            feedback_count = 0;

        end else begin

            if (
                dut.u_bridge.shaf_rvalid &&
                dut.u_bridge.shaf_rready
            ) begin

                $display(
                    "SHA_WORD %0d %08x %x",
                    sha_word_count,
                    dut.u_bridge.shaf_rdata.data,
                    dut.u_bridge.shaf_rdata.mask
                );

                sha_word_count = sha_word_count + 1;

            end


            if (dut.hmac_fifo_push) begin

                $display(
                    "FEEDBACK %0d SEL=%0d DATA=%08x",
                    feedback_count,
                    dut.hmac_fifo_wdata_sel,
                    dut.hmac_fifo_data
                );

                feedback_count = feedback_count + 1;

            end

        end
    end


    initial begin : run_test

        integer cycles;

        rst_ni = 1'b0;
        start_i = 1'b0;

        key_i =
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;

        message_i =
            256'h000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f;


        repeat (5) @(posedge clk_i);

        #1 rst_ni = 1'b1;

        repeat (3) @(posedge clk_i);

        #1 start_i = 1'b1;

        @(posedge clk_i);

        #1 start_i = 1'b0;


        cycles = 0;

        while (!done_o && cycles < 20000) begin
            @(posedge clk_i);
            #1;
            cycles++;
        end


        $display("TOTAL_SHA_WORDS=%0d", sha_word_count);
        $display("TOTAL_FEEDBACK=%0d", feedback_count);
        $display("FINAL_DIGEST=%064x", digest_o);

        if (!done_o) begin
            $display("TRACE_RESULT=TIMEOUT");
        end else begin
            $display("TRACE_RESULT=COMPLETE");
        end

        $finish;

    end

endmodule
