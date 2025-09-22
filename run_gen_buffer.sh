set -e
echo "Starting buffer generation..."
# loop over games in a for loop: pong, beamrider, enduro, spaceinvaders, asteroids, centipede and py-asteroids
for game in pong beamrider enduro spaceinvaders asteroids centipede py-asteroids; do
    echo "Generating buffer for $game..."
    uv run generate_dqn_buffer.py $game --size 100000
done
echo "All buffers generated."
