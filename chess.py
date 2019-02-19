import numpy as np
import random
import sys
import format_tools as fmt
from copy import deepcopy


"""
TODO:
    - pawn promotion
    - check and checkmate test
    - draw if no captures for n moves or known piece combinations
    - en passant working? no en passant in Move.__repr__
    - fix piece reps to be more sensible
"""

"""
Search for bookmarks by pattern:
    - (Piece)
    - (Board) 
    - (Player)
    - (Controller)
    - (Test)  
    - (Main)  
"""


# ------------------------------------------------------------------------------------------------------
#                                                                                                (Piece) 
# ------------------------------------------------------------------------------------------------------


class Piece:
    def __init__(self, color):
        self.color = color
        self.rep = None
        self.value = 0
        self.name = 'none'
    
    def __repr__(self):
        return self.rep


class Pawn(Piece):
    """
    move_history:
        - 0 if pawn hasn't moved (two square move available)
        - 1 if pawn just moved (can be taken by en passant if moved two squares)
        - 2 or greater otherwise (no special rules)
    """
    def __init__(self, color):
        super().__init__(color)
        self.dir = -1 if color == 'w' else 1
        self.move_history = 0
        self.moved_two_squares = False
        self.rep = color + 'p'
        self.value = 1
        self.name = 'pawn'


class Knight(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.rep = color + 'n'
        self.value = 3
        self.paths = (
            ( 1,  2),
            ( 1, -2),
            (-1,  2),
            (-1, -2),
            ( 2,  1),
            ( 2, -1),
            (-2,  1),
            (-2, -1)
        )
        self.name = 'knight'


class Bishop(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.rep = color + 'b'
        self.value = 3
        self.paths = (
            ( 1,   1),
            ( 1,  -1),
            (-1,   1),
            (-1 , -1)
        )
        self.name = 'bishop'


class Rook(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.rep = color + 'r'
        self.value = 5
        self.paths = (
            ( 0,  1),
            ( 1,  0),
            ( 0, -1),
            (-1,  0)
        )
        self.name = 'rook'


class Queen(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.rep = color + 'q'
        self.value = 9
        self.paths = (
            ( 0,   1),
            ( 1,   0),
            ( 0,  -1),
            (-1,   0),
            ( 1,   1),
            ( 1,  -1),
            (-1,   1),
            (-1,  -1)
        )
        self.name = 'queen'


class King(Piece):
    def __init__(self, color):
        super().__init__(color)
        self.rep = color + 'k'
        self.value = 9999999
        self.paths = (
            ( 0,   1),
            ( 1,   0),
            ( 0,  -1),
            (-1,   0),
            ( 1,   1),
            ( 1,  -1),
            (-1,   1),
            (-1,  -1)
        )
        self.name = 'king'


# ------------------------------------------------------------------------------------------------------
#                                                                                                (Board)
# ------------------------------------------------------------------------------------------------------


_rtrans = {0:8, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1} 
_ctrans = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}


class Board:
    """
    Search for bookmarks by pattern:
        - (Move)
        - (BoardDunder) 
        - (BoardInit)
        - (BoardMove)
        - (BoardSquare)
        - (BoardPiece)  
    """

# ---------------------------------------------------------------------------------- (Move)

    class Move:
        def __init__(self, piece, old_pos, new_pos, capture, en_passant_cap=None):
            self.piece = piece
            self.old_pos = old_pos
            self.new_pos = new_pos
            self.capture = capture
            self.en_passant_cap = en_passant_cap

        def __repr__(self):
            return ', '.join([
                self.piece.rep, 
                f'({_ctrans[self.old_pos[1]]}{_rtrans[self.old_pos[0]]})',
                f'({_ctrans[self.new_pos[1]]}{_rtrans[self.new_pos[0]]})',
                'nocap' if self.capture is None else self.capture.rep
            ])

# ---------------------------------------------------------------------------------- (BoardDunder)

    def __init__(self):
        self.kings = dict()
        self.board, self.piece_to_pos = self.board_init()
        self.move_stack = list()
        self.movesets = {'w': list(), 'b': list()}
        self.def_piece_squares = {'w': list(), 'b': list()}
        self.in_check = list()
        self.promote_piece = None

    def __repr__(self):
        rows = [f'{i}' for i in range(8, 0, -1)]
        cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        horizontal_line = '   ' + ('-' * 81) + '\n'
        light_fill = '|' + '\u2588' * 9
        dark_fill = '|         '
        rep = ''
        
        for i in range(8):
            light_squares = range(0, 9, 2) if i % 2 == 0 else range(1, 9, 2)

            fill = [light_fill if square in light_squares else dark_fill for square in range(8)]
            fill_line = '   ' + ''.join(fill) + '|\n'

            piece_reps = list()
            for j in range(8):
                if j in light_squares:
                    piece_reps.append(
                        f'{light_fill[:3]}({self.board[i, j].rep[0]}|{self.board[i, j].rep[1]})' +
                        f'{light_fill[8:]}' \
                        if self.board[i, j] is not None else light_fill
                    )
                else:
                    piece_reps.append(
                        f'{dark_fill[:3]}({self.board[i, j].rep[0]}|{self.board[i, j].rep[1]})' +
                        f'{dark_fill[8:]}' \
                        if self.board[i, j] is not None else dark_fill
                    )
            piece_line = f' {rows[i]} ' + ''.join(piece_reps) + '|\n'

            rep += horizontal_line
            rep += fill_line 
            rep += piece_line
            rep += fill_line 

        rep += horizontal_line + '\n'
        rep += '        ' + '         '.join(cols) + '\n\n'

        return rep

# ---------------------------------------------------------------------------------- (BoardInit)

    def board_init(self):
        board = np.array([[None for i in range(8)] for j in range(8)])
        self.pieces_init(board, Pawn, 'b', (1,), tuple(range(8)))
        self.pieces_init(board, Pawn, 'w', (6,), tuple(range(8)))
        self.pieces_init(board, Knight, 'b', (0,), (1, 6))
        self.pieces_init(board, Knight, 'w', (7,), (1, 6))
        self.pieces_init(board, Bishop, 'b', (0,), (2, 5))
        self.pieces_init(board, Bishop, 'w', (7,), (2, 5))
        self.pieces_init(board, Rook, 'b', (0,), (0, 7))
        self.pieces_init(board, Rook, 'w', (7,), (0, 7))
        self.pieces_init(board, Queen, 'b', (0,), (3,))
        self.pieces_init(board, Queen, 'w', (7,), (3,))
        self.pieces_init(board, King, 'b', (0,), (4,))
        self.pieces_init(board, King, 'w', (7,), (4,))

        piece_to_pos = dict()
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] is not None:
                    piece_to_pos[board[i, j]] = (i, j)

        return board, piece_to_pos
    
    def pieces_init(self, board, piece_type, color, rows, cols):
        for row in rows:
            for col in cols:
                piece = piece_type(color)
                board[row, col] = piece
        if isinstance(piece, King):
            self.kings[color] = piece

# ---------------------------------------------------------------------------------- (BoardSquare)

    @staticmethod
    def square_inbounds(square):
        return 0 <= square[0] <= 7 and 0 <= square[1] <= 7

    def square_empty(self, pos):
        return self.board[pos[0], pos[1]] == None
    
    def square_friendly(self, pos, color):
        return not self.square_empty(pos) and self.board[pos[0], pos[1]].color == color
    
    def square_opponent(self, pos, color):
        return not self.square_empty(pos) and self.board[pos[0], pos[1]].color != color
    
    def piece_to_square(self, piece):
        return self.piece_to_pos[piece]
    
    def square_to_piece(self, pos):
        return self.board[pos[0], pos[1]]

# ---------------------------------------------------------------------------------- (BoardMove)

    def get_color_pieces(self, color):
        pieces = list()
        for row in self.board:
            for piece in row:
                if piece is not None and piece.color == color:
                    pieces.append(piece)
        return pieces

    def color_moves(self, color):
        return self.movesets[color]

    def push_move(self, move):
        self.move_stack.append(move)

        for piece in self.board:
            if isinstance(piece, Pawn) and piece.move_history > 0:
                piece.move_history += 1
                if piece.move_history >= 2 and piece.moved_two_squares:
                    piece.moved_two_squares = False
                if move.new_pos[0] == 0 or move.new_pos[0] == 7:
                    self.promote_piece = piece 

        if isinstance(move.piece, Pawn) and move.piece.move_history == 0:
            move.piece.move_history = 1
            if abs(move.new_pos[0] - move.old_pos[0]) == 2:
                move.piece.moved_two_squares = True

        if move.en_passant_cap is not None:
            en_passant_piece = move.en_passant_cap[0]
            en_passant_pos = move.en_passant_cap[1]
            self.board[en_passant_pos[0], en_passant_pos[1]] = None
            del self.piece_to_pos[en_passant_piece]
        elif move.capture is not None:
            del self.piece_to_pos[move.capture]
        if move.en_passant_cap is None:
            self.board[move.new_pos[0], move.new_pos[1]] = move.piece

        self.board[move.old_pos[0], move.old_pos[1]] = None
        self.piece_to_pos[move.piece] = move.new_pos
    
    def pop_move(self):
        move = self.move_stack.pop()
        
        if move.new_pos == move.old_pos:
            self.demote_pawn(move.piece)            
        else:
            self.board[move.old_pos[0], move.old_pos[1]] = move.piece
            self.piece_to_pos[move.piece] = move.old_pos

            if isinstance(move.piece, Pawn) and move.piece.moved_two_squares:
                move.piece.moved_two_squares = False
            if move.en_passant_cap is None:
                self.board[move.new_pos[0], move.new_pos[1]] = move.capture
            if move.en_passant_cap is not None: 
                en_passant_piece = move.en_passant_cap[0]
                en_passant_pos = move.en_passant_cap[1]
                self.board[en_passant_pos[0], en_passant_pos[1]] = en_passant_piece
                self.piece_to_pos[en_passant_piece] = en_passant_pos
            elif move.capture is not None:
                self.piece_to_pos[move.capture] = move.new_pos

        for row in self.board:
            for piece in row:
                if isinstance(piece, Pawn):
                    if piece.move_history > 0:
                        piece.move_history -= 1
                    if piece.move_history == 1 and 4 <= self.piece_to_square(piece)[0] <= 5:
                        piece.moved_two_squares = True

    def refresh_color_moves(self, color, king_search):
        """
        Generates moveset for one side. 
        king_search: T/F whether to get king move information from king_get_moves or get_king_area
        """
        moves = list()
        defended_squares = list()
        checks = list()
        king_moves = None

        for piece in self.get_color_pieces(color):
            if isinstance(piece, Pawn):
                movement, defended_piece_squares = self.pawn_get_moves(piece)
            elif isinstance(piece, Knight):
                movement, defended_piece_squares = self.knight_get_moves(piece)
            elif isinstance(piece, Bishop) \
            or isinstance(piece, Rook) \
            or isinstance(piece, Queen):
                movement, defended_piece_squares = self.ranger_get_moves(piece)
            elif isinstance(piece, King):
                if king_search:
                    movement, defended_piece_squares, checks = self.king_get_moves(piece)
                    king_moves = movement
                else:
                    movement, defended_piece_squares = self.get_king_area(piece)
            moves.extend(movement)
            defended_squares.extend(defended_piece_squares)

        if king_search:
            checks_len = len(checks)
            if checks_len > 0:
                if checks_len > 1:
                    moves = king_moves
                else:
                    pass
                    check_res_path = self.check_resolution_path(moves, color, checks)
                    illegal_moves = list()
                    for move in moves:
                        if move.new_pos not in check_res_path and not isinstance(move.piece, King):
                            illegal_moves.append(move)
                    for move in illegal_moves:
                        moves.remove(move)
        
        return moves, defended_squares, checks
    
    def check_resolution_path(self, moves, color, checks):
        """
        Isolates the path of check that pieces other than the king can use to
        block or capture and resolve the check.
        """
        king_square = self.piece_to_square(self.kings[color])
        opp_piece_type = type(checks[0].piece)
        opp_piece_square = checks[0].old_pos
        path = [opp_piece_square]

        if opp_piece_type == Queen or opp_piece_type == Rook or opp_piece_type == Bishop: 
            dir_row = np.sign(king_square[0] - opp_piece_square[0])
            dir_col = np.sign(king_square[1] - opp_piece_square[1])
            while king_square != opp_piece_square:
                path.append(opp_piece_square)
                opp_piece_square = (opp_piece_square[0] + dir_row, opp_piece_square[1] + dir_col)
        return path

    def refresh_moves(self, cur_color):
        opp_color = 'w' if cur_color == 'b' else 'b'

        self.movesets[opp_color], self.def_piece_squares[opp_color], self.checks  = \
            self.refresh_color_moves(opp_color, False)
        self.movesets[cur_color], self.def_piece_squares[cur_color], self.checks = \
            self.refresh_color_moves(cur_color, True)
   
    def build_move(self, piece, new_pos, en_passant=False):
        if en_passant:
            ep_cap = self.square_to_piece(new_pos)
            actual_new_pos = (new_pos[0] + piece.dir, new_pos[1])
            move = self.Move(
                piece,
                self.piece_to_square(piece),
                actual_new_pos,
                None,
                (ep_cap, new_pos)
            )
        else:
            move = self.Move(
                piece, 
                self.piece_to_square(piece), 
                new_pos,
                self.square_to_piece(new_pos)
            )
        return move

# ---------------------------------------------------------------------------------- (BoardPiece)

    def pawn_get_moves(self, pawn):
        moves = list()
        pawn_pos = self.piece_to_square(pawn)
        one_square = (pawn_pos[0] + pawn.dir, pawn_pos[1])
        two_squares = (pawn_pos[0] + (2 * pawn.dir), pawn_pos[1])
        cap_left = (pawn_pos[0] + pawn.dir, pawn_pos[1] - 1)
        cap_right = (pawn_pos[0] + pawn.dir, pawn_pos[1] + 1)
        en_passant_left = (pawn_pos[0], pawn_pos[1] - 1)
        en_passant_right = (pawn_pos[0], pawn_pos[1] + 1)
        defended_piece_squares = list()

        if self.square_inbounds(one_square) and self.square_empty(one_square):
            moves.append(self.build_move(pawn, one_square))
            if self.square_inbounds(two_squares) \
            and pawn.move_history == 0 \
            and self.square_empty(two_squares):
                moves.append(self.build_move(pawn, two_squares))
        if self.square_inbounds(cap_left): 
            if self.square_opponent(cap_left, pawn.color):
                moves.append(self.build_move(pawn, cap_left))
            elif self.square_friendly(cap_left, pawn.color):
                defended_piece_squares.append(cap_left)
        if self.square_inbounds(cap_right):
            if self.square_opponent(cap_right, pawn.color):
                moves.append(self.build_move(pawn, cap_right))
            elif self.square_friendly(cap_right, pawn.color):
                defended_piece_squares.append(cap_right)
        if self.square_inbounds(en_passant_left):
            enp_piece_left = self.square_to_piece(en_passant_left)
            if isinstance(enp_piece_left, Pawn) \
            and enp_piece_left.move_history == 1 \
            and enp_piece_left.moved_two_squares == True:
                moves.append(self.build_move(pawn, en_passant_left, True))
        if self.square_inbounds(en_passant_right):
            enp_piece_right = self.square_to_piece(en_passant_right)
            if isinstance(enp_piece_right, Pawn) \
            and enp_piece_right.move_history == 1 \
            and enp_piece_right.moved_two_squares == True:
                moves.append(self.build_move(pawn, en_passant_right, True))

        return moves, defended_piece_squares
    
    def pawn_promote(self, pawn, new_type):
        new_piece = new_type(pawn.color)
        square = self.piece_to_square(pawn)
        self.board[square[0], square[1]] = new_piece
        self.piece_to_pos[new_piece] = square
        del self.piece_to_pos[pawn]

    def pawn_demote(self, piece):
        pawn = Pawn(piece.color)
        square = self.piece_to_square(piece)
        self.board[square[0], square[1]] = pawn
        self.piece_to_pos[pawn] = square
        del self.piece_to_pos[piece]
    
    def knight_get_moves(self, knight):
        moves = list()
        knight_pos = self.piece_to_square(knight)
        defended_piece_squares = list()
        
        for path in knight.paths:
            move_square = (knight_pos[0] + path[0], knight_pos[1] + path[1])
            if self.square_inbounds(move_square):
                if self.square_empty(move_square) or self.square_opponent(move_square, knight.color):
                    moves.append(self.build_move(knight, move_square))
                elif self.square_friendly(move_square, knight.color):
                    defended_piece_squares.append(move_square)

        return moves, defended_piece_squares
    
    def ranger_get_moves(self, ranger):
        """
        Find moves for a queen, bishop, or rook
        """
        moves = list()
        ranger_pos = self.piece_to_square(ranger)
        defended_piece_squares = list()

        for path in ranger.paths:
            move = ranger_pos
            while True:
                move = (move[0] + path[0], move[1] + path[1])
                if self.square_inbounds(move):
                    if self.square_empty(move):
                        moves.append(self.build_move(ranger, move))
                    elif self.square_opponent(move, ranger.color):
                        moves.append(self.build_move(ranger, move))
                        break
                    else:
                        defended_piece_squares.append(move)
                        break
                else:
                    break

        return moves, defended_piece_squares

    def king_get_moves(self, king):
        opp_color = 'w' if king.color == 'b' else 'b'
        opp_player_moves, opp_def_pc_squares = \
            self.movesets[opp_color], self.def_piece_squares[opp_color]
        opp_player_moves = [ 
            move.new_pos for move in opp_player_moves \
            if not isinstance(move.piece, Pawn) \
            or move.new_pos[1] != move.old_pos[1] 
        ]

        king_pos = self.piece_to_square(king)
        moves = [(king_pos[0] + path[0], king_pos[1] + path[1]) for path in king.paths]

        defended_piece_squares = [
            square for square in moves \
            if self.square_inbounds(square) \
            and self.square_friendly(square, king.color)
        ]
        moves = [
            self.build_move(king, new_pos) for new_pos in moves \
            if self.square_inbounds(new_pos) \
            and new_pos not in opp_player_moves \
            and new_pos not in opp_def_pc_squares \
            and not self.square_friendly(new_pos, king.color)
        ]
        checks = [
            move for move in self.movesets[opp_color] \
            if king_pos == move.new_pos
            and (
                not isinstance(move.piece, Pawn)
                or move.new_pos[1] != move.old_pos[1]
            )
        ]

        return moves, defended_piece_squares, checks
    
    def get_king_area(self, king):
        """
        King movement is dependent on the opposing king's position, since they can't attack each other.
        With that in mind, infinite recursion can easily become a problem when looking for a king's
        available moves. To avoid this, instead of the king (of the player whose turn it is)
        being aware of where the other king can move legally, it respects the opposing king's 
        whole movement square.
        """
        if king in self.piece_to_pos:
            king_pos = self.piece_to_pos[king]
            moves = [(king_pos[0] + path[0], king_pos[1] + path[1]) for path in king.paths]
            defended_piece_squares = [
                square for square in moves 
                if self.square_inbounds(square) and self.square_friendly(square, king.color)
            ]

            return (
                [self.build_move(king, move) for move in moves if self.square_inbounds(move)],
                defended_piece_squares
            )
        else:
            return [list(), list()]


# ------------------------------------------------------------------------------------------------------
#                                                                                               (Player)
# ------------------------------------------------------------------------------------------------------


class Player:
    def __init__(self, color, board):
        self.board = board
        self.color = color
        self.moves = None
        self.opponent = None
    
    def know_opponent(self, opponent):
        self.opponent = opponent
    
    def know_moves(self, moves):
        self.moves = moves


class HumanPlayer(Player):
    def __init__(self, color, board):
        super().__init__(color, board)


class AIPlayer(Player):
    def __init__(self, color, board):
        super().__init__(color, board)


# ------------------------------------------------------------------------------------------------------
#                                                                                           (Controller)
# ------------------------------------------------------------------------------------------------------


class Controller:
    def __init__(self):
        pass    
    
    def players_init(player_types, board):
        pass
    
    def make_move(from_square, to_square, resign=False, offer_draw=False):
        if resign:
            pass
        elif offer_draw:
            pass
        else:
            pass

    def available_moves():
        pass


# ------------------------------------------------------------------------------------------------------
#                                                                                                 (Test)
# ------------------------------------------------------------------------------------------------------


def run_tests():
    print('\n-------------\nTest Summary:\n-------------')
    random_movement_test(200, 200, True)


def make_error_record(piece, pos, error_type, funcname):
    if error_type == 'missing':
        return f'{funcname}(): piece {piece} was missing from square {pos}'
    elif error_type == 'misplaced':
        return f'{funcname}(): piece {piece} was misplaced on square {pos}'
    else:
        return '{funcname}(): ({piece}, {pos}) -- an error was recorded but the type is unknown'


def random_movement_test(max_moves, iterations, iter_output=False):
    """
    Pushes (max_moves or less) random moves onto the board's move stack, pops the number of moves 
    made, and makes sure the pieces are in their starting positions with verify_starting_pos()
    """
    error_record = list()
    test_passed = True
    board = Board()

    for test_iter in range(iterations):
        cur_color = 'w'
        opp_color = 'b'
        moves_made = 0
        for i in range(max_moves):
            board.refresh_moves(cur_color)
            moves = board.color_moves(cur_color)
            if len(moves) > 0:
                random_move = moves[random.randint(0, len(moves) - 1)]
                board.push_move(random_move)
                opp_color, cur_color = cur_color, opp_color
                moves_made += 1
            else:
                break
            # input(board)
        for i in range(moves_made):
            board.pop_move()
            # input(board)
        verify_starting_pos(board, error_record)
        if len(error_record) > 0:
            test_passed = False
            if iter_output:
                print(f'random_movement_test(): iteration {test_iter} errors:')
                for error in error_record:
                    print(f'\t{error}') 
                cont = input('\ncontinue random movement test? (y/n): ')
                if 'n' in cont.lower():
                    break
            else:
                break
        error_record.clear()

    print(f'random_movement_test(): {"pass" if test_passed else "fail"}')


def verify_starting_pos(board, error_record):
    """
    Verifies that all pieces are in starting positions. Tests by piece type, not instance.
    Good for testing the board's move stack by pushing, then popping n moves from a clean board,
    assuming instances of the same type don't get mixed around.
    """
    row_colors = ['b', 'b', None, None, None, None, 'w', 'w']
    for row in range(8):
        for col in range(8):
            piece = board.square_to_piece((row, col))
            if row == 0 or row == 7:
                if col == 0 or col == 7: 
                    piece_type = Rook
                elif col == 1 or col == 6:
                    piece_type = Knight
                elif col == 2 or col == 5:
                    piece_type = Bishop
                elif col == 3:
                    piece_type = Queen
                else: 
                    piece_type = King
            elif row == 1 or row == 6:
                piece_type = Pawn
            else:
                piece_type = None
            if piece_type is not None and not isinstance(piece, piece_type):
                error_record.append(
                    make_error_record(piece_type.__name__, (row, col), 'missing', 'verify_starting_pos'))
            if ( \
                row_colors[row] is not None \
                and piece is not None \
                and piece.color is not row_colors[row] 
            ) \
            or row_colors[row] is None \
            and piece is not None:
                error_record.append(make_error_record(piece.name, (row, col), 'misplaced')) 


# ------------------------------------------------------------------------------------------------------
#                                                                                                 (Main)
# ------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    try: 
        assert len(sys.argv) > 2
        assert sys.argv[1] == 'rt'
        for param in sys.argv[2:]:
            if param == '-all':
                run_tests() 
            if param == '-rand':
                random_movement_test(200, 100, True)
    except AssertionError:
        test_strs = [
            'all tests: "python chess.py rt -all"', 
            'random movement test: "python chess.py rt -rand"'
        ]
        formatted_strs = fmt.alignOnPattern('"', test_strs)
        print(  
            '\nchess.py is a chess game interface.\n\n' 
            + 'To run tests, try one of the following:\n'  
            + '---------------------------------------' 
        )
        for string in formatted_strs:
            print(string)
    except Exception as e:
        print(e)
    
