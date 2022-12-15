import chess
import torch

PIECE_MAP = {
    'k': 0,
    'q': 1,
    'r': 2,
    'n': 3,
    'b': 4,
    'p': 5,
}


class Chess:
    def __init__(self, max_moves=50):
        self.board = chess.Board()
        self.pieces_map = PIECE_MAP
        self.max_moves = max_moves
        self.current_move = 0

    def encode_board(self):
        pieces = self.board.epd().split(' ', 1)[0]
        rows = pieces.split('/')
        torch_board = torch.zeros((12, 8, 8))
        for i, row in enumerate(rows):
            j = 0
            for item in row:
                if item.isdigit():
                    j += int(item)
                else:
                    piece = self.pieces_map[item.lower()]
                    if item.islower():
                        piece += 6
                    torch_board[piece][i][j] = 1
                    j += 1
        return torch_board

    def step(self, move):
        self.board.push_san(move)
        reward, done = self.game_over()
        return self.encode_board(), reward, done

    def reset(self):
        self.board = chess.Board()
        self.current_move = 0
        return self.encode_board()

    def get_legal_moves(self):
        return [self.board.san(move) for move in self.board.legal_moves]

    def game_over(self):
        outcome = self.board.outcome(claim_draw=True)
        self.current_move += 1
        if outcome is None and self.current_move < self.max_moves:
            return 0, False
        elif outcome is None and self.current_move >= self.max_moves:
            return -.5, True
        elif outcome.termination == chess.Termination.CHECKMATE:
            return 1, True
        else:
            return 0, True


if __name__ == '__main__':
    env = Chess()
    state, _,_ = env.step('e4')
    print(state[11])
