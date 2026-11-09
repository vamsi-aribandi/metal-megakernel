#pragma once
// Shared constants, prepended to the shader source at runtime.
#define VOCAB 96
#define CONTEXT 8
#define EMBED 16
#define INPUTS (CONTEXT * EMBED)
#define HIDDEN 128
#define W_EMBED 0
#define W_HIDDEN (VOCAB * EMBED)
#define W_HBIAS (W_HIDDEN + HIDDEN * INPUTS)
#define W_OUTPUT (W_HBIAS + HIDDEN)
#define W_OBIAS (W_OUTPUT + VOCAB * HIDDEN)
#define PARAMS (W_OBIAS + VOCAB)
#define WIDTH 320
#define HEIGHT 200
#define TOKEN_CAP 4096
// CPU-owned input buffer. GPU only reads.
#define C_STOP 0
#define C_KEYS 1
#define C_TIME_MS 2
#define C_MAX_STEPS 3
#define C_SEED 4
#define C_WORDS 32
// GPU-owned output buffer. CPU only reads. Each role owns whole cache lines.
#define O_STEPS 0
#define O_LOSS 1
#define O_FIRST_LOSS 2
#define O_TRAIN_DONE 3
#define O_SNAPSHOT_VERSION 4
#define O_TRAIN_SEES_TOKENS 5
#define O_TRAIN_SEES_FRAMES 6
#define O_TOKENS 32
#define O_INFER_VERSION 33
#define O_INFER_DONE 34
#define O_INFER_SEES_STEPS 35
#define O_INFER_SEES_FRAMES 36
#define O_FRAMES 64
#define O_GAME_DONE 69
#define O_GAME_SEES_STEPS 70
#define O_GAME_SEES_TOKENS 71
#define O_TRAIN_HASH 7
#define O_INFER_HASH 37
#define O_SNAPSHOT_ERRORS 38
#define O_SNAPSHOT_COPIES 39
#define O_TOKEN_BASE 128
#define O_TOKEN_VERSION_BASE (O_TOKEN_BASE + TOKEN_CAP)
#define O_WORDS (O_TOKEN_VERSION_BASE + TOKEN_CAP)
// Keys: forward/backward/left/right/strafe-left/strafe-right.
#define KEY_FORWARD 1
#define KEY_BACKWARD 2
#define KEY_LEFT 4
#define KEY_RIGHT 8
#define KEY_STRAFE_LEFT 16
#define KEY_STRAFE_RIGHT 32
