import matplotlib.pyplot as plt


def draw(p: list, color: str, alpha=1.0):
    if len(p) == 1:
        x, y, r = p[0][0], p[0][1], p[0][2]
        plt.Circle((x, y), r, color=color, alpha=alpha, fill=True)
    if len(p) == 2:
        x, y = [p[0][0], p[1][0], p[1][0], p[0][0]], [p[0][1], p[0][1], p[1][1], p[1][1]]
        plt.fill(x, y, color, alpha=alpha)
    if len(p) > 2:
        x, y = [point[0] for point in p], [point[1] for point in p]
        plt.fill(x, y, color, alpha=alpha)


def draw_ui(knight: dict, enemies: list, attacks: list = None):
    plt.cla()
    plt.xlim((0, 2400))
    plt.ylim((0, 1080))

    draw(knight['position'], '#0000FF', 0.5)

    for enemy in enemies:
        draw(enemy['position'], '#FF0000', 0.5)

    for attack in attacks:
        draw(attack['position'], '#00FFFF', 0.1)

    plt.pause(0.05)


def start_draw():
    plt.ion()


def end_draw():
    plt.ioff()


if __name__ == '__main__':
    draw_ui({'position': [[748, 228], [715, 143]]},
            [
                {'position': [[870, 314], [930, 143]]},
                # {'position': [[1081, 195], [1078, 180], [960, 183]]},  # Hit GDash, 武器1
                # {'position': [[911, 730], [901, 717], [982, 631]]},  # Hit ADash, 武器2, 从高空冲刺的时候
                # Hit Counter 1, 攻击1, 反击或者格挡
                # {'position': [[996, 246], [1003, 336], [922, 436], [787, 228], [779, 136], [898, 162]]},
                # Hit Counter 2, 攻击2, 反击或者格挡
                # {'position': [[857, 454], [755, 463], [601, 412], [493, 278], [463, 133], [712, 134]]},
            ],
            [
                # WallSlash, 未出现
                # {'position': [[843, -461], [913, -505], [926, -561], [894, -595], [722, -629], [713, -431]]},
                # Slash, 攻击(打到了), 可以计算出y轴方向上的攻击距离
                {'position': [[922, 322], [1018, 260], [1037, 181], [982, 140], [743, 110], [731, 357]]},
                # DownSlash, 下劈, 可以计算出y轴方向上的攻击距离
                # {'position': [[1144,194],[1161,17],[1216,-72],[1325,-75],[1391,34],[1423,198]]},
                # AltSlash, 攻击, 可以计算出x轴方向上的攻击距离
                # {'position': [[869, 337], [1006, 282], [1036, 196], [975, 135], [769, 114], [754, 351]]},
                # UpSlash, 向上攻击, 可以计算出y轴方向上的攻击距离
                # {'position': [[1325,408],[1253,410],[1206,336],[1201,190],[1387,193],[1371,331]]},
            ]
            )
