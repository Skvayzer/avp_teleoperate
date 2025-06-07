from vuer import Vuer, VuerSession
from vuer.schemas import Hands #, MotionControllers
from asyncio import sleep

app = Vuer(host='0.0.0.0', port=8012, ngrok=True)#, cert='./vuer/cert.pem', key='./vuer/key.pem')

# app.add_handler("CONTROLLER_MOVE")(self.on_motion_controller_move)


@app.add_handler("HAND_MOVE")
async def handler(event, session):
    print(f"Hand Movement Event: key-{event.key}", event.value)

# @app.add_handler("CANERA_MOVE")
# async def camera_handler(event, session):
#     print(f"Camera Movement Event: key-{event.key}", event.value)

# @app.add_handler("*")
# async def handler(event, session):
#     # print("a")
#     print(f"блять key-{event.key}", event.value)

# @app.add_handler("CONTROLLER_MOVE")
# async def handler(event, session):
#     # print("a")
#     print(f"Movement Event: key-{event.key}", event.value)


@app.spawn(start=True)
async def main(session: VuerSession):
    # Important: You need to set the `stream` option to `True` to start
    # streaming the hand movement.
    # session.upsert @ MotionControllers(stream=True, key="motion-controller", left=True, right=True)

    session.upsert(
        Hands(
            fps=60, stream=True, key="hands", showLeft=True, showRight=True
            # hideLeft=True,       # hides the hand, but still streams the data.
            # hideRight=True,      # hides the hand, but still streams the data.
            # disableLeft=True,    # disables the left data stream, also hides the hand.
            # disableRight=True,   # disables the right data stream, also hides the hand.
        ),
        to="bgChildren",
    )

    # session.upsert @ MotionControllers(stream=True, key="hands")
    
    while True:
        await sleep(1)