import chess
import torch
from QNetwork import QNetwork

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
        torch_board = torch.zeros((6, 8, 8))
        for i, row in enumerate(rows):
            j = 0
            for item in row:
                if item.isdigit():
                    j += int(item)
                else:
                    if item.isupper():
                        torch_board[self.pieces_map[item.lower()]][i][j] = 1
                    else:
                        torch_board[self.pieces_map[item]][i][j] = -1
                    j += 1
        return torch_board

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
    # env = Chess()
    # env.step('e4')
    # board = env.encode_board()
    # model = QNetwork()
    # print(model(board).reshape(128))
