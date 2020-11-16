import pygame as pg
import random as rd
import os
import neat

#Colours
black       = (0,0,0)
white       = (255,255,255)
red         = (255, 0, 0)

# Window size
xmax = 500  # pixels
ymax = 200  # pixels

# Time step
dt = 0.01

# initial conditions
score = 0
highscore = 0
gen = 0


class Block:
    WIDTH = 20
    HEIGHT = 20
    COLOR = black

    def __init__(self, x, y):
        self.vy = 0
        self.x = x
        self.y = (ymax - y)
        self.h = self.y
        self.passed = False


    def draw(self, scr):
        ay = 1000

        self.vy += ay * dt
        self.y += self.vy * dt

        if self.y >= self.h:
            self.y = self.h
            self.vy = 0

        self.block = pg.Rect(self.x, self.y - self.HEIGHT, self.WIDTH, self.HEIGHT)
        pg.draw.rect(scr, self.COLOR, self.block, 5)

    def jump(self):
        if self.y == self.h:
            self.vy -= 250


class Obstacle(Block):
    WIDTH = 10
    HEIGHT = 10
    COLOR = red
    vx = rd.randrange(-800,-400)

    def draw(self, scr):
        global score
        self.x += self.vx * dt

        self.obstacle = pg.Rect(self.x, self.y - self.HEIGHT, self.WIDTH, self.HEIGHT)
        pg.draw.rect(scr, self.COLOR, self.obstacle)

        if self.obstacle.right <= 0:
            self.vx = rd.randrange(-800,-400)
            self.x = xmax
            score += 1

class Ground:
    COLOR = black

    def __init__(self, x, h, w):
        self.w = w
        self.h = h
        self.x = x
        self.y = ymax - self.h

    def draw(self, scr):
        ground = pg.Rect(self.x, self.y, self.w, self.h)
        pg.draw.rect(scr, self.COLOR, ground)

def collide(block, obstacle):
    if pg.Rect.colliderect(block.block, obstacle.obstacle):
        return True
    return False

def passed(block, obstacle):
    if not block.passed and block.x > obstacle.x:
        return True
    return False


def write_text(txt, x_pos, y_pos, scr, font=15, color=black):
    font = pg.font.Font('./Minecraft.ttf', font)
    text = font.render(txt, True, color)
    Rect = text.get_rect()
    Rect.center = (x_pos, y_pos)
    scr.blit(text, Rect)



def play(genomes, config):
    global gen
    global score
    global highscore

    gen += 1
    nets = []
    ge = []
    blocks = []
    score = 0

    # Initialise pygame
    pg.init()

    # Create window
    scr = pg.display.set_mode((xmax, ymax))
    pg.display.set_caption('NEAT')

    # Create objects
    ground = Ground(0, ymax/4, xmax)
    obstacle = Obstacle(xmax, ground.h)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        blocks.append(Block(xmax/3, ground.h))
        g.fitness = 0
        ge.append(g)

    playing = True
    while playing:
        # Event pump
        pg.event.pump()

        # Clear screen
        scr.fill(white)
        ground.draw(scr)
        obstacle.draw(scr)

        # Draw text
        write_text(f"Generation: {gen}", 60, 20, scr)
        write_text(f"Number of blocks: {len(blocks)}", xmax - 80, 20, scr)
        write_text(f"Score: {score}", xmax/2, 20, scr)
        write_text(f"Highscore: {highscore}", xmax / 2, 40, scr)


        if len(blocks) == 0:
            break

        # Update highscore
        if highscore < score:
            highscore = score

        for x, block in enumerate(blocks):
            block.draw(scr)
            ge[x].fitness += 0.1
            output = nets[x].activate((block.x, obstacle.x))

            if output[0] > 0.5:
                block.jump()
                ge[x].fitness -= 2

            if passed(block, obstacle):
                ge[x].fitness += 5

            if collide(block, obstacle):
                ge[x].fitness -= 5
                blocks.pop(x)
                nets.pop(x)
                ge.pop(x)


        # events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                playing = False
                # Close window
                pg.quit()

        # Update screen
        pg.display.flip()



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # play for x generations
    winner = p.run(play, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)