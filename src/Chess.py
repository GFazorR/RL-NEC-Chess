import chess
import numpy as np

PIECE_MAP = {
    'k': 0,
    'q': 1,
    'r': 2,
    'n': 3,
    'b': 4,
    'p': 5,
}


class Chess:
    def __init__(self):
        self.board = chess.Board()
        self.pieces_map = PIECE_MAP

    def encode_board(self):
        pieces = self.board.epd().split(' ', 1)[0]
        rows = pieces.split('/')
        np_board = np.zeros((64,6))
        for i, row in enumerate(rows):
            j = i*8
            for item in row:
                if item.isdigit():
                    j += int(item)
                else:
                    if item.isupper():
                        np_board[j][self.pieces_map[item.lower()]] = 1
                    else:
                        np_board[j][self.pieces_map[item]] = -1
                    j += 1
        return np_board.reshape((64*6))

    def step(self, move):
        if self.board.parse_san(move) not in self.board.legal_moves:
            raise Exception("Illegal Move")
        self.board.push_san(move)
        reward, done = self.game_over()
        return self.encode_board(), reward, done

    def reset(self):
        self.board = chess.Board()

    def get_legal_moves(self):
        return self.board.legal_moves

    # TODO Define action space
    def get_action_space(self):
        pass

    def game_over(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return 0, False
        elif outcome.termination == 1:
            return 1, True
        else:
            return .5, True


if __name__ == '__main__':
    pass


