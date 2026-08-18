/**
 * ------------------------------------------------------------
 * Copyright (c) All rights reserved 
 * SiLab, Institute of Physics, University of Bonn
 * ------------------------------------------------------------
 */
 
`timescale 1ps / 1ps
//`default_nettype none

module pixcap65 (

    input wire FCLK_IN, 

    //full speed 
    inout wire [7:0] BUS_DATA,
    input wire [15:0] ADD,
    input wire RD_B,
    input wire WR_B,

    //high speed
    inout wire [7:0] FD,
    input wire FREAD,
    input wire FSTROBE,
    input wire FMODE,

    //debug
    output wire LED1,
    output wire LED2,
    output wire LED3,
    output wire LED4,
    output wire LED5,

    inout SDA,
    inout SCL,

    //SRAM
    output wire [19:0] SRAM_A,
    inout wire [15:0] SRAM_IO,
    output wire SRAM_BHE_B,
    output wire SRAM_BLE_B,
    output wire SRAM_CE1_B,
    output wire SRAM_OE_B,
    output wire SRAM_WE_B,
		
		// PIXCAP65 chip
		output wire RST_B,
		output wire SDI,
		output wire SCK,
		output wire LOAD,
		output wire CLK0,
		output wire CLK1,
		output wire CLK2,
		output wire CLK3,
		
		// misc
		input wire LCK1,
		output wire MULTI_IO_0,
		
		input wire SDO

);   
    parameter ABUSWIDTH = 16;

    assign  MULTI_IO_0 = LCK1;

    assign SDA = 1'bz;
    assign SCL = 1'bz;

    
    (* KEEP = "{TRUE}" *) 
    wire CLK320;
    (* KEEP = "{TRUE}" *) 
    wire CLK160;
    (* KEEP = "{TRUE}" *) 
    wire BUS_CLK;
    (* KEEP = "{TRUE}" *) 
    wire SPI_CLK;
    (* KEEP = "{TRUE}" *) 
    wire TDC_WCLK; 
    
    wire CLK_LOCKED;
    wire BUS_RST;

    reset_gen i_reset_gen(.CLK(BUS_CLK), .RST(BUS_RST));

    clk_gen i_clkgen(
        .CLKIN(FCLK_IN),
        .BUS_CLK(BUS_CLK),
        .U2_CLK5(),
        .U2_CLK80(TDC_WCLK),
        .U2_CLK160(CLK160),
        .U2_CLK320(CLK320),
        .SPI_CLK(SPI_CLK),
        .LOCKED(CLK_LOCKED)
    ); 
		
		reg [3:0]sck_cnt;
		wire SPI_CLK_DIV;
		wire LCK1_BUF;
		
		always @(posedge SPI_CLK)
		begin
		  sck_cnt <= sck_cnt + 1;
		end
		
		
    BUFG SPI_CLK_DIV_BUFG_INST (.I(sck_cnt[2]), .O(SPI_CLK_DIV));
    BUFG LCK1_BUFG_INST (.I(LCK1), .O(LCK1_BUF));
	  

 // -------  MODULE ADREESSES  ------- //
    localparam GPIO_BASEADDR = 16'h0000;
    localparam GPIO_HIGHADDR = 16'h000f;

    localparam SEQ_GEN_BASEADDR = 16'h1000;                      //0x1000
    localparam SEQ_GEN_HIGHADDR = SEQ_GEN_BASEADDR + 16 + 16'h1fff;   //0x300f

    localparam SPI_BASEADDR = 16'h4000;                      //0x1000
    localparam SPI_HIGHADDR = SPI_BASEADDR + 16 + 16'h1fff;   //0x300f

    
 // -------  BUS SIGNALING  ------- //
    wire [15:0] BUS_ADD;
    wire BUS_RD, BUS_WR;
    fx2_to_bus i_fx2_to_bus (
        .ADD(ADD),
        .RD_B(RD_B),
        .WR_B(WR_B),

        .BUS_CLK(BUS_CLK),
        .BUS_ADD(BUS_ADD),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
        .CS_FPGA()
    );

 // -------  USER MODULES  ------- //
       
    wire [3:0] SEQ_OUT;
    seq_gen 
    #( 
        .BASEADDR(SEQ_GEN_BASEADDR), 
        .HIGHADDR(SEQ_GEN_HIGHADDR),
        .MEM_BYTES(8*1024), 
        .OUT_BITS(8) 
    ) i_seq_gen
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
    
        .SEQ_CLK(LCK1_BUF),
        .SEQ_OUT(SEQ_OUT)
    
    );
    
    assign CLK0 = SEQ_OUT[0];
    assign CLK1 = SEQ_OUT[1];   
    assign CLK2 = SEQ_OUT[2];   
    assign CLK3 = SEQ_OUT[3];     
		
		
		
    spi
    #(
        .BASEADDR(SPI_BASEADDR),
        .HIGHADDR(SPI_HIGHADDR),
        .ABUSWIDTH(ABUSWIDTH),
        .MEM_BYTES(4096) 
    )  i_spi
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA[7:0]),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
        
        .SPI_CLK(SPI_CLK_DIV),
        .EXT_START(),
        
        .SCLK(SCK),
        .SDI(SDI),
        .SDO(SDO),
        .SEN(),
        .SLD(LOAD)
    );
    
   
    gpio 
    #( 
        .BASEADDR(GPIO_BASEADDR), 
        .HIGHADDR(GPIO_HIGHADDR),
        .IO_WIDTH(8),
        .IO_DIRECTION(8'hff)
    ) i_gpio
    (
        .BUS_CLK(BUS_CLK),
        .BUS_RST(BUS_RST),
        .BUS_ADD(BUS_ADD),
        .BUS_DATA(BUS_DATA),
        .BUS_RD(BUS_RD),
        .BUS_WR(BUS_WR),
        .IO({LED5, LED4, LED3, LED2, LED1, RST_B})
    );
     
endmodule
