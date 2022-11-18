import unittest

import torch

from NeuralEpisodicControl.Chess import Chess
import chess


def create_board():
    board_actual = torch.zeros((6, 8, 8))
    # kings
    board_actual[0][0][4] = -1
    board_actual[0][7][4] = 1
    # queens
    board_actual[1][0][3] = -1
    board_actual[1][7][3] = 1
    # rooks
    board_actual[2][0][[0, 7]] = -1
    board_actual[2][7][[0, 7]] = 1
    # knights
    board_actual[3][0][[1, 6]] = -1
    board_actual[3][7][[1, 6]] = 1
    # bishops
    board_actual[4][0][[2, 5]] = -1
    board_actual[4][7][[2, 5]] = 1
    # pawns
    board_actual[5][1] = torch.ones(8) * -1
    board_actual[5][6] = torch.ones(8)
    return board_actual


class MyTestCase(unittest.TestCase):
    # TODO Modify
    def test_reset(self):
        chess_test = Chess()
        chess_actual = chess.Board()

        chess_test.step('e4')
        self.assertNotEqual(chess_test.board, chess_actual)
        chess_test.reset()
        self.assertEqual(chess_test.board, chess_actual)

    def test_encode_board(self):
        board_test = Chess()
        board_actual = create_board()

        test = board_test.encode_board()

        self.assertEqual(test.shape, board_actual.shape)
        self.assertEqual(type(test), type(board_actual))
        self.assertTrue(torch.equal(test, board_actual))

        board_test.step('e4')
        test = board_test.encode_board()

        board_actual[5][6][4] = 0
        board_actual[5][4][4] = 1

        self.assertEqual(test.shape, board_actual.shape)
        self.assertEqual(type(test), type(board_actual))
        self.assertTrue(torch.equal(test, board_actual))

    def test_get_legal_moves(self):
        chess_test = Chess()
        chess_actual = chess.Board()
        self.assertEqual(list(chess_test.get_legal_moves()),
                         list(chess_actual.legal_moves))
        chess_test.step('e4')
        self.assertNotEqual(list(chess_test.get_legal_moves()),
                            list(chess_actual.legal_moves))
        chess_actual.push_san('e4')
        self.assertEqual(list(chess_test.get_legal_moves()),
                         list(chess_actual.legal_moves))

    def test_step(self):
        chess_test = Chess()
        chess_actual = chess.Board()
        board_actual = create_board()
        test, reward, done = chess_test.step('e4')

        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(test.shape, board_actual.shape)
        self.assertEqual(type(test), type(board_actual))
        self.assertFalse(torch.equal(test, board_actual))

        board_actual[5][6][4] = 0
        board_actual[5][4][4] = 1

        self.assertEqual(reward, 0)
        self.assertFalse(done)
        self.assertEqual(test.shape, board_actual.shape)
        self.assertEqual(type(test), type(board_actual))
        self.assertTrue(torch.equal(test, board_actual))

    def test_game_over(self):
        chess_test = Chess()
        chess_test.step('e4')
        reward, done = chess_test.game_over()
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        chess_test.board = chess.Board('r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4')
        reward, done = chess_test.game_over()
        self.assertEqual(reward, 1)
        self.assertTrue(done)
        chess_test.board = chess.Board('4k3/8/8/8/8/8/8/4K3 b KQkq - 0 4')
        reward, done = chess_test.game_over()
        self.assertEqual(reward, .5)
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
