%matplotlib inline
from matplotlib import pyplot as plt

import desolver as de
import desolver.backend as D

D.set_float_fmt("float64")
# %%


def Fij(ri, rj, G):
    rel_r = rj - ri
    return G * (1 / D.norm(rel_r, ord=2) ** 3) * rel_r


def rhs(t, state, masses, G):
    total_acc = D.zeros_like(state)

    for idx, (ri, mi) in enumerate(zip(state, masses)):
        for jdx, (rj, mj) in enumerate(zip(state[idx + 1 :], masses[idx + 1 :])):
            partial_force = Fij(ri[:3], rj[:3], G)
            total_acc[idx, 3:] += partial_force * mj
            total_acc[idx + jdx + 1, 3:] -= partial_force * mi

    total_acc[:, :3] = state[:, 3:]

    return total_acc


# %%
Msun = 1.98847 * 10 ** 30  ## Mass of the Sun, kg
AU = 149597871e3  ## 1 Astronomical Unit, m
year = 365.25 * 24 * 3600  ## 1 year, s
G = 4 * D.pi ** 2  ## in solar masses, AU, years
V = D.sqrt(
    G
)  ## Speed scale corresponding to the orbital speed required for a circular orbit at 1AU with a period of 1yr

# %%
initial_state = D.array(
    [
        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.25, 0.9682458365518543, 0.0, 0.9682458365518543 * 0, -0.25 * 0, 0.0],
        [-0.5, -0.8660254037844386, 0.0, -0.8660254037844386 * 0, 0.5 * 0, 0.0],
    ]
)

masses = D.array(
    [
        1,
        1,
        1,
        1,
    ]
)

rhs(0.0, initial_state, masses, G)
# %%
a = de.OdeSystem(
    rhs,
    y0=initial_state,
    dense_output=True,
    t=(0, 2.0),
    dt=0.00001,
    rtol=1e-14,
    atol=1e-14,
    constants=dict(G=G, masses=masses),
)
a.method = "RK1412"

a.integrate()

# %%

fig = plt.figure(figsize=(16, 16))

com_motion = D.sum(a.y[:, :, :] * masses[None, :, None], axis=1) / D.sum(masses)

fig = plt.figure(figsize=(16, 16))
ax1 = fig.add_subplot(131, aspect=1)
ax2 = fig.add_subplot(132, aspect=1)
ax3 = fig.add_subplot(133, aspect=1)

ax1.set_xlabel("x (AU)")
ax1.set_ylabel("y (AU)")
ax2.set_xlabel("y (AU)")
ax2.set_ylabel("z (AU)")
ax3.set_xlabel("z (AU)")
ax3.set_ylabel("x (AU)")

for i in range(a.y.shape[1]):
    ax1.plot(a.y[:, i, 0], a.y[:, i, 1], color=f"C{i}")
    ax2.plot(a.y[:, i, 1], a.y[:, i, 2], color=f"C{i}")
    ax3.plot(a.y[:, i, 2], a.y[:, i, 0], color=f"C{i}")

ax1.scatter(com_motion[:, 0], com_motion[:, 1], color="k")
ax2.scatter(com_motion[:, 1], com_motion[:, 2], color="k")
ax3.scatter(com_motion[:, 2], com_motion[:, 0], color="k")

plt.tight_layout()

# %%


from matplotlib import animation, rc

# set to location of ffmpeg to get animations working
# For Linux or Mac
# plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

# For Windows
plt.rcParams[
    "animation.ffmpeg_path"
] = "C:\\Users\\Igor Strozzi\\Documents\\ffmpeg\\bin\\ffmpeg.exe"

from IPython.display import HTML

# %%

%%capture

# This magic command prevents the creation of a static figure image so that we can view the animation in the next cell
t = a.t
all_states = a.y
planets = [all_states[:, i, :] for i in range(all_states.shape[1])]
com_motion = D.sum(all_states * masses[None, :, None], axis=1) / D.sum(masses)

plt.ioff()

fig = plt.figure(figsize=(16, 8))
ax1 = fig.add_subplot(131, aspect=1)
ax2 = fig.add_subplot(132, aspect=1)
ax3 = fig.add_subplot(133, aspect=1)

ax1.set_xlabel("x (AU)")
ax1.set_ylabel("y (AU)")
ax2.set_xlabel("y (AU)")
ax2.set_ylabel("z (AU)")
ax3.set_xlabel("z (AU)")
ax3.set_ylabel("x (AU)")

xlims = D.abs(a.y[:, :, 0]).max()
ylims = D.abs(a.y[:, :, 1]).max()
zlims = D.abs(a.y[:, :, 2]).max()

ax1.set_xlim(-xlims - 0.25, xlims + 0.25)
ax2.set_xlim(-ylims - 0.25, ylims + 0.25)
ax3.set_xlim(-zlims - 0.25, zlims + 0.25)

ax1.set_ylim(-ylims - 0.25, ylims + 0.25)
ax2.set_ylim(-zlims - 0.25, zlims + 0.25)
ax3.set_ylim(-xlims - 0.25, xlims + 0.25)

planets_pos_xy = []
planets_pos_yz = []
planets_pos_zx = []

planets_xy = []
planets_yz = []
planets_zx = []

(com_xy,) = ax1.plot(
    [], [], color="k", linestyle="", marker="o", markersize=5.0, zorder=10
)
(com_yz,) = ax2.plot(
    [], [], color="k", linestyle="", marker="o", markersize=5.0, zorder=10
)
(com_zx,) = ax3.plot(
    [], [], color="k", linestyle="", marker="o", markersize=5.0, zorder=10
)

event_counter = 0

close_encounter_xy = []
close_encounter_yz = []
close_encounter_zx = []

for i in range(len(planets)):
    close_encounter_xy.append(
        ax1.plot(
            [], [], color=f"k", marker="x", markersize=3.0, linestyle="", zorder=9
        )[0]
    )
    close_encounter_yz.append(
        ax2.plot(
            [], [], color=f"k", marker="x", markersize=3.0, linestyle="", zorder=9
        )[0]
    )
    close_encounter_zx.append(
        ax3.plot(
            [], [], color=f"k", marker="x", markersize=3.0, linestyle="", zorder=9
        )[0]
    )

for i in range(a.y.shape[1]):
    planets_xy.append(ax1.plot([], [], color=f"C{i}", zorder=8)[0])
    planets_yz.append(ax2.plot([], [], color=f"C{i}", zorder=8)[0])
    planets_zx.append(ax3.plot([], [], color=f"C{i}", zorder=8)[0])
    planets_pos_xy.append(
        ax1.plot([], [], color=f"C{i}", linestyle="", marker=".", zorder=8)[0]
    )
    planets_pos_yz.append(
        ax2.plot([], [], color=f"C{i}", linestyle="", marker=".", zorder=8)[0]
    )
    planets_pos_zx.append(
        ax3.plot([], [], color=f"C{i}", linestyle="", marker=".", zorder=8)[0]
    )


def init():
    global event_counter
    for i in range(len(planets)):
        planets_xy[i].set_data([], [])
        planets_yz[i].set_data([], [])
        planets_zx[i].set_data([], [])
        planets_pos_xy[i].set_data([], [])
        planets_pos_yz[i].set_data([], [])
        planets_pos_zx[i].set_data([], [])

    com_xy.set_data([], [])
    com_yz.set_data([], [])
    com_zx.set_data([], [])

    for i in range(len(planets)):
        close_encounter_xy[i].set_data(
            a.events[event_counter].y[i, 0], a.events[event_counter].y[i, 1]
        )
        close_encounter_yz[i].set_data(
            a.events[event_counter].y[i, 1], a.events[event_counter].y[i, 2]
        )
        close_encounter_zx[i].set_data(
            a.events[event_counter].y[i, 2], a.events[event_counter].y[i, 0]
        )

    return tuple(
        planets_xy
        + planets_yz
        + planets_zx
        + planets_pos_xy
        + planets_pos_yz
        + planets_pos_zx
        + [com_xy, com_yz, com_zx]
        + [close_encounter_xy, close_encounter_yz, close_encounter_zx]
    )


def animate(frame_num):
    global event_counter
    for i in range(len(planets)):
        planets_xy[i].set_data(
            planets[i][max(frame_num - 5, 0) : frame_num, 0],
            planets[i][max(frame_num - 5, 0) : frame_num, 1],
        )
        planets_yz[i].set_data(
            planets[i][max(frame_num - 5, 0) : frame_num, 1],
            planets[i][max(frame_num - 5, 0) : frame_num, 2],
        )
        planets_zx[i].set_data(
            planets[i][max(frame_num - 5, 0) : frame_num, 2],
            planets[i][max(frame_num - 5, 0) : frame_num, 0],
        )
        planets_pos_xy[i].set_data(
            planets[i][frame_num : frame_num + 1, 0],
            planets[i][frame_num : frame_num + 1, 1],
        )
        planets_pos_yz[i].set_data(
            planets[i][frame_num : frame_num + 1, 1],
            planets[i][frame_num : frame_num + 1, 2],
        )
        planets_pos_zx[i].set_data(
            planets[i][frame_num : frame_num + 1, 2],
            planets[i][frame_num : frame_num + 1, 0],
        )

    com_xy.set_data(
        com_motion[frame_num : frame_num + 1, 0],
        com_motion[frame_num : frame_num + 1, 1],
    )
    com_yz.set_data(
        com_motion[frame_num : frame_num + 1, 1],
        com_motion[frame_num : frame_num + 1, 2],
    )
    com_zx.set_data(
        com_motion[frame_num : frame_num + 1, 2],
        com_motion[frame_num : frame_num + 1, 0],
    )

    if t[frame_num] >= a.events[event_counter].t and event_counter + 1 < len(a.events):
        event_counter += 1
        for i in range(len(planets)):
            close_encounter_xy[i].set_data(
                a.events[event_counter].y[i, 0], a.events[event_counter].y[i, 1]
            )
            close_encounter_yz[i].set_data(
                a.events[event_counter].y[i, 1], a.events[event_counter].y[i, 2]
            )
            close_encounter_zx[i].set_data(
                a.events[event_counter].y[i, 2], a.events[event_counter].y[i, 0]
            )

    return tuple(
        planets_xy
        + planets_yz
        + planets_zx
        + planets_pos_xy
        + planets_pos_yz
        + planets_pos_zx
        + [com_xy, com_yz, com_zx]
        + [close_encounter_xy, close_encounter_yz, close_encounter_zx]
    )


ani = animation.FuncAnimation(
    fig,
    animate,
    list(range(1, len(t))),
    interval=1500.0 / 60.0,
    blit=False,
    init_func=init,
)

rc("animation", html="html5")

# Uncomment to save an mp4 video of the animation

ani.save("Nbodies.mp4", fps=30)
