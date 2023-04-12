


class Env():
    def __init__(self, max_bv_num: int) -> None:
        self.state_dim = (max_bv_num + 1) * 4   # x_pos, y_pos, speed, yaw
        self.action_dim = 2                     # delta_speed, delta_yaw
        self.action_range = [-6, 2, -30, 30]
