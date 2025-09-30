# Asteroids AI Project Development TODO list


## Tasks
Context: ROADMAP 3.3.2 Add versioning for Py-Asteroids game setup  

currently I use py-asteroids as the game name for the custom py-asteroids game
To avoid confusion when further investigating effects of game setup on the difficulty for agents, I want to have exact versioning on the setups.
The important configurations are set in asteroids/entities/constants.py.
they are imported by other scripts of the asteroids game.
I want to be able to be able to feeze current setup/configuration as py-asteroids-v1, and increase the version when I need to update or extend the game.
- [x] find a way to organize game versioning (ex constants_v1.py  constants_v2.py and let other import by version at run time according to version?) 
- [x] I have create a variable in py_asteroids_name_id_map


## Backlog

- [ ] review v1.0 experiments results
    - [ ] evaluate reward
    - [ ] compare hyperparameters
    - [ ] record best video for demo
- [ ] py-asteroids game version and action mapping
- [ ] fix print/log messages for noisy network (not epsilon but other better metrics)
- [ ] Add time information


## Reminder

TODO is about: 
- How and When
- Story and Task level