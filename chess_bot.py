
















import chess
import random
import time

# Piece-square tables (simplified)
piece_square_table = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5,
        1, 1, 2, 3, 3, 2, 1, 1,
        0.5, 0.5, 1, 2.5, 2.5, 1, 0.5, 0.5,
        0, 0, 0, 2, 2, 0, 0, 0,
        0.5, -0.5, -1, 0, 0, -1, -0.5, 0.5,
        0.5, 1, 1, -2, -2, 1, 1, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -5, -4, -3, -3, -3, -3, -4, -5,
        -4, -2, 0, 0, 0, 0, -2, -4,
        -3, 0, 1, 1.5, 1.5, 1, 0, -3,
        -3, 0.5, 1.5, 2, 2, 1.5, 0.5, -3,
        -3, 0, 1.5, 2, 2, 1.5, 0, -3,
        -3, 0.5, 1, 1.5, 1.5, 1, 0.5, -3,
        -4, -2, 0, 0.5, 0.5, 0, -2, -4,
        -5, -4, -3, -3, -3, -3, -4, -5
    ],
    chess.BISHOP: [
        -2, -1, -1, -1, -1, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 1, 1, 0.5, 0, -1,
        -1, 0.5, 0.5, 1, 1, 0.5, 0.5, -1,
        -1, 0, 1, 1, 1, 1, 0, -1,
        -1, 1, 1, 1, 1, 1, 1, -1,
        -1, 0.5, 0, 0, 0, 0, 0.5, -1,
        -2, -1, -1, -1, -1, -1, -1, -2
    ],
    chess.ROOK: [
        0, 0, 0, 0.5, 0.5, 0, 0, 0,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        -0.5, 0, 0, 0, 0, 0, 0, -0.5,
        0.5, 1, 1, 1, 1, 1, 1, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.QUEEN: [
        -2, -1, -1, -0.5, -0.5, -1, -1, -2,
        -1, 0, 0, 0, 0, 0, 0, -1,
        -1, 0, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -0.5, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        0, 0, 0.5, 0.5, 0.5, 0.5, 0, -0.5,
        -1, 0.5, 0.5, 0.5, 0.5, 0.5, 0, -1,
        -1, 0, 0.5, 0, 0, 0, 0, -1,
        -2, -1, -1, -0.5, -0.5, -1, -1, -2
    ],
    chess.KING: [
        0, 0, 0, 0, 0, 0, 0, 0,
        0.5, 1, 1, 1, 1, 1, 1, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
}

piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Custom board printing with Unicode pieces and labels
def print_board_with_symbols(board):
    symbols = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
    }

    board_rows = board.board_fen().split('/')
    print("    a b c d e f g h")
    print("  +-----------------+")
    for rank_index, row in enumerate(board_rows):
        display_row = ""
        for ch in row:
            if ch.isdigit():
                display_row += ". " * int(ch)
            else:
                display_row += f"{symbols.get(ch, ch)} "
        print(f"{8 - rank_index} | {display_row}| {8 - rank_index}")
    print("  +-----------------+")
    print("    a b c d e f g h")


def evaluate_board(board):
    score = 0
    for piece_type in piece_values:
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)
        score += len(white_pieces) * piece_values[piece_type]
        score -= len(black_pieces) * piece_values[piece_type]
        for sq in white_pieces:
            score += piece_square_table[piece_type][sq]
        for sq in black_pieces:
            score -= piece_square_table[piece_type][chess.square_mirror(sq)]
    return score

def alpha_beta(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)

    legal_moves = list(board.legal_moves)
    if maximizing:
        max_eval = -float('inf')
        for move in legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    best_move = None
    best_eval = -float('inf') if board.turn == chess.WHITE else float('inf')

    for move in board.legal_moves:
        board.push(move)
        eval = alpha_beta(board, depth - 1, -float('inf'), float('inf'), board.turn == chess.BLACK)
        board.pop()
        if board.turn == chess.WHITE and eval > best_eval:
            best_eval = eval
            best_move = move
        elif board.turn == chess.BLACK and eval < best_eval:
            best_eval = eval
            best_move = move
    return best_move

def play_game():
    board = chess.Board()
    while not board.is_game_over():
        print_board_with_symbols(board)
        print()
        if board.turn == chess.WHITE:
            move = input("Enter your move (UCI, e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move.")
            except:
                print("Invalid input.")
        else:
            print("Bot thinking...")
            start = time.time()
            bot_move = find_best_move(board, depth=4)
            end = time.time()
            print(f"Bot move: {bot_move}, computed in {end - start:.2f} seconds.")
            board.push(bot_move)

    print_board_with_symbols(board)
    print("Game Over. Result:", board.result())

if __name__ == "__main__":
    play_game()
