#pragma once
// Minimal 5x7 bitmap font, hand-drawn (not copied from any font file) - just enough glyphs
// for the debug panel's own text. Each row's bits read left-to-right as bit4..bit0, so a
// row literally reads as 0bXXXXX in the same left-to-right order it's drawn.
#include <cstdint>

struct Glyph {
    uint8_t rows[7];
};

inline const Glyph* getGlyph(char c) {
    static const Glyph g0 = {{0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110}};
    static const Glyph g1 = {{0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110}};
    static const Glyph g2 = {{0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111}};
    static const Glyph g3 = {{0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110}};
    static const Glyph g4 = {{0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010}};
    static const Glyph g5 = {{0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110}};
    static const Glyph g6 = {{0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110}};
    static const Glyph g7 = {{0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000}};
    static const Glyph g8 = {{0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110}};
    static const Glyph g9 = {{0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100}};

    static const Glyph gA = {{0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001}};
    static const Glyph gB = {{0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110}};
    static const Glyph gC = {{0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111}};
    static const Glyph gD = {{0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110}};
    static const Glyph gE = {{0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111}};
    static const Glyph gF = {{0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000}};
    static const Glyph gG = {{0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111}};
    static const Glyph gH = {{0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001}};
    static const Glyph gI = {{0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110}};
    static const Glyph gL = {{0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111}};
    static const Glyph gM = {{0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001}};
    static const Glyph gN = {{0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001}};
    static const Glyph gO = {{0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110}};
    static const Glyph gP = {{0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000}};
    static const Glyph gR = {{0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001}};
    static const Glyph gS = {{0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110}};
    static const Glyph gT = {{0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100}};
    static const Glyph gU = {{0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110}};
    static const Glyph gX = {{0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001}};
    static const Glyph gY = {{0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100}};
    static const Glyph gZ = {{0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111}};

    static const Glyph gColon = {{0b00000, 0b00100, 0b00000, 0b00000, 0b00100, 0b00000, 0b00000}};
    static const Glyph gPeriod = {{0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100}};
    static const Glyph gMinus = {{0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000}};
    static const Glyph gSpace = {{0, 0, 0, 0, 0, 0, 0}};

    switch (c) {
        case '0': return &g0; case '1': return &g1; case '2': return &g2;
        case '3': return &g3; case '4': return &g4; case '5': return &g5;
        case '6': return &g6; case '7': return &g7; case '8': return &g8;
        case '9': return &g9;
        case 'A': return &gA; case 'B': return &gB; case 'C': return &gC;
        case 'D': return &gD; case 'E': return &gE; case 'F': return &gF;
        case 'G': return &gG; case 'H': return &gH; case 'I': return &gI; case 'L': return &gL;
        case 'M': return &gM; case 'N': return &gN; case 'O': return &gO;
        case 'P': return &gP; case 'R': return &gR; case 'S': return &gS;
        case 'T': return &gT; case 'U': return &gU; case 'X': return &gX;
        case 'Y': return &gY; case 'Z': return &gZ;
        case ':': return &gColon; case '.': return &gPeriod; case '-': return &gMinus;
        default: return &gSpace;
    }
}
