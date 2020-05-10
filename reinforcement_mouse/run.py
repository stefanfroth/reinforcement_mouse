import cv2
from mouse import Mouse
from grid import Grid

mouse = Mouse()
grid = Grid(mouse)

while True:
    mouse.update_values()

    if mouse.state in [1, 9]:
        mouse.state = 5
        mouse.last_state = 5
        mouse.last_reward = 0


    print(f'The Q-values of the mouse are \n {mouse.q}')
    print(f'The mouse is currently in state {mouse.state}')


    grid.draw()

    cv2.imshow('frame', grid.frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    mouse.choose_action()
    mouse.assign_reward()
    print(f'The mouse chooses action {mouse.action}/{mouse.action_mapping[mouse.action]}')
    time.sleep(2)

    # input_ = input('Press "c" to continue')


cv2.destroyAllWindows()
