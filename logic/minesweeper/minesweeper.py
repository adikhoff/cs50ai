import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if (self.count == len(self.cells)):
            return self.cells
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if (self.count == 0):
            return self.cells
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if (cell in self.cells):
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if (cell in self.cells):
            self.cells.remove(cell)
    
    def is_subset_of(self, other):
        is_subset = True
        for cell in self.cells:
            if cell not in other.cells:
                is_subset = False
        return is_subset


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
            
    def neighbors(self, cell):
        neighbors = set()
        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds
                if 0 <= i < self.height and 0 <= j < self.width:
                    fcell = (i, j)
                    neighbors.add(fcell)
        return neighbors
    
    def print_knowledge(self):
        for sentence in self.knowledge:
            print(f"  {sentence}")
        print(f"turn: {len(self.moves_made)}, safesleft: {len(self.safes) - len(self.moves_made)}, safes: {len(self.safes)}, mines: {len(self.mines)}")
                    
    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        
        # 2) mark the cell as safe
        self.mark_safe(cell)
        
        # 3) add a new sentence to the AI's knowledge base
        #    based on the value of `cell` and `count`
        neighbors = self.neighbors(cell)
        sentence = Sentence(neighbors, count)
        for cell in self.mines:
            sentence.mark_mine(cell)
        for cell in self.safes:
            sentence.mark_safe(cell)
        self.knowledge.append(sentence)
        
        sweeping = True
        while (sweeping):
            sweeping = False
            # 4) mark any additional cells as safe or as mines
            #    if it can be concluded based on the AI's knowledge base
            safes = set()
            mines = set()
            for sentence in self.knowledge:
                for cell in sentence.known_safes():
                    safes.add(cell)
                for cell in sentence.known_mines():
                    mines.add(cell)
            for cell in safes:
                sweeping = True
                self.mark_safe(cell)
            for cell in mines:
                sweeping = True
                self.mark_mine(cell)
            for sentence in self.findEmptySentences():
                sweeping = True
                self.knowledge.remove(sentence)
            # 5) add any new sentences to the AI's knowledge base
            #    if they can be inferred from existing knowledge
            inferences = self.inferSubsets()
            for sentence in inferences:
                if sentence not in self.knowledge:
                    sweeping = True
                    self.knowledge.append(sentence)
        self.print_knowledge()

    def inferSubsets(self):
        newSentences = []
        for sentence1 in self.knowledge:
            for sentence2 in self.knowledge:
                if (sentence1 == sentence2):
                    continue
                if (sentence1.is_subset_of(sentence2)):
                    newCells = []
                    for cell in sentence2.cells:
                        if cell not in sentence1.cells:
                            newCells.append(cell)
                    newSentence = Sentence(newCells, sentence2.count - sentence1.count)
                    newSentences.append(newSentence)
        return newSentences

    def findEmptySentences(self):
        emptySentences = []
        for sentence in self.knowledge:
            if (len(sentence.cells) == 0):
                emptySentences.append(sentence)
        return emptySentences
        
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for move in self.safes:
            if move not in self.moves_made:
                return move
        return None

    def make_mark_mine_move(self):
        for mine in self.mines:
            if mine not in self.moves_made:
                self.moves_made.add(mine)
                return mine
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        bestCells = []
        lowrisk = float("inf")
        for i in range(0, self.height):
            for j in range(0, self.width):
                cell = (i, j)
                if cell not in self.moves_made and cell not in self.mines:
                    cellScore = 0
                    for sentence in self.knowledge:
                        if cell in sentence.cells:
                            cellScore = max(cellScore, sentence.count)
                    if (cellScore == lowrisk):
                        bestCells.append(cell)
                    if (cellScore < lowrisk):
                        lowrisk = cellScore
                        bestCells = []
                        bestCells.append(cell)
        if (len(bestCells) == 0):
            return None
        bestCell = bestCells[random.randint(0, len(bestCells) - 1)]
        return bestCell
                
