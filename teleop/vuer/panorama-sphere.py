# from asyncio import sleep
# import numpy as np

# from vuer import Vuer, VuerSession
# from vuer.schemas import DefaultScene, Arrow, Sphere

# app = Vuer(host='0.0.0.0', port=8012, cert='../cert.pem', key='../key.pem')

# n = 10
# N = 1000

# sphere = Sphere(
#     args=[1, 32, 32],
#     materialType="standard",
#     material={"map": "./farm_house.jpg", "side": 1},
#     position=[0, 0, 0],
#     rotation=[0.5 * np.pi, 0, 0],
# )


# @app.spawn(start=True)
# async def main(proxy: VuerSession):
#     proxy.set @ DefaultScene(sphere)

#     # keep the main session alive.
#     while True:
#         await sleep(1)


from asyncio import sleep

from vuer import Vuer
from vuer.events import Set
from vuer.schemas import DefaultScene, Sphere


app = Vuer(host='0.0.0.0', port=8012, cert='../cert.pem', key='../key.pem')

# use `start=True` to start the app immediately
@app.spawn(start=True)
async def main(session):
    session @ Set(
        DefaultScene(
            # SceneBackground(),
            Sphere(
                key="ball",
                args=[1, 20, 20],
                position=[0, 0.5, 0],
                materialType="standard",
                material=dict(
                    map="https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png",
                    transparent=True,
                    side=1,
                ),
            ),
            up=[0, 1, 0],
        ),
    )

    while True:
        await sleep(0.016)