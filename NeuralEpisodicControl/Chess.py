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
    def __init__(self):
        self.board = chess.Board()
        self.pieces_map = PIECE_MAP

    def encode_board(self, play_as_white):
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
        if not play_as_white:
            torch_board = -torch_board
            torch_board = torch.flip(torch_board, (-1, 1))

        return torch_board

    def step(self, move, play_as_white):
        self.board.push_san(move)
        reward, done = self.game_over()
        return self.encode_board(play_as_white), reward, done

    def reset(self):
        self.board = chess.Board()
        return self.encode_board(True)

    def get_legal_moves(self):
        return [self.board.san(move) for move in self.board.legal_moves]

    def game_over(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return 0, False
        elif outcome.termination == chess.Termination.CHECKMATE:
            return 1, True
        else:
            return 0, True


if __name__ == '__main__':
    pass
