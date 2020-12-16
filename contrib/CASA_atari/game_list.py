import atari_py

# [(0, 'Alien-v0'),
#  (1, 'Amidar-v0'),
#  (2, 'Assault-v0'),
#  (3, 'Asterix-v0'),
#  (4, 'Asteroids-v0'),
#  (5, 'Atlantis-v0'),
#  (6, 'BankHeist-v0'),
#  (7, 'BattleZone-v0'),
#  (8, 'BeamRider-v0'),
#  (9, 'Berzerk-v0'),
#  (10, 'Bowling-v0'),
#  (11, 'Boxing-v0'),
#  (12, 'Breakout-v0'),
#  (13, 'Centipede-v0'),
#  (14, 'ChopperCommand-v0'),
#  (15, 'CrazyClimber-v0'),
#  (16, 'Defender-v0'),
#  (17, 'DemonAttack-v0'),
#  (18, 'DoubleDunk-v0'),
#  (19, 'Enduro-v0'),
#  (20, 'FishingDerby-v0'),
#  (21, 'Freeway-v0'),
#  (22, 'Frostbite-v0'),
#  (23, 'Gopher-v0'),
#  (24, 'Gravitar-v0'),
#  (25, 'Hero-v0'),
#  (26, 'IceHockey-v0'),
#  (27, 'Jamesbond-v0'),
#  (28, 'Kangaroo-v0'),
#  (29, 'Krull-v0'),
#  (30, 'KungFuMaster-v0'),
#  (31, 'MontezumaRevenge-v0'),
#  (32, 'MsPacman-v0'),
#  (33, 'NameThisGame-v0'),
#  (34, 'Phoenix-v0'),
#  (35, 'Pitfall-v0'),
#  (36, 'Pong-v0'),
#  (37, 'PrivateEye-v0'),
#  (38, 'Qbert-v0'),
#  (39, 'Riverraid-v0'),
#  (40, 'RoadRunner-v0'),
#  (41, 'Robotank-v0'),
#  (42, 'Seaquest-v0'),
#  (43, 'Skiing-v0'),
#  (44, 'Solaris-v0'),
#  (45, 'SpaceInvaders-v0'),
#  (46, 'StarGunner-v0'),
#  (47, 'Tennis-v0'),
#  (48, 'TimePilot-v0'),
#  (49, 'Tutankham-v0'),
#  (50, 'UpNDown-v0'),
#  (51, 'Venture-v0'),
#  (52, 'VideoPinball-v0'),
#  (53, 'WizardOfWor-v0'),
#  (54, 'YarsRevenge-v0'),
#  (55, 'Zaxxon-v0')]


def upper_1st(s):
    res = ""
    up = False
    for t in "_" + s:
        if t == "_":
            up = True
        else:
            if up:
                res += t.upper()
                up = False
            else:
                res += t
    return res


def get_games():
    not_in = ['adventure',
              'air_raid',
              'carnival',
              'elevator_action',
              'journey_escape',
              'kaboom',
              'pooyan', ]
    games = atari_py.list_games()

    games = list(set(games) - set(not_in))

    games.sort()
    games = [upper_1st(game) + "-v0" for game in games]
    return games
