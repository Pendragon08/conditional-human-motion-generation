# Training visualization gate

New formal reinforcement-learning runs must have a locally visible, continuously updating view of the **actual training simulation** before launch. Validate the server GPU renderer, streaming connection, robot and scene visibility with a short preflight; record the observed result. Run training and monitoring as independent processes so closing or losing the local viewer does not stop learning. A separate checkpoint replay may aid diagnosis, but it is not the live training view and cannot satisfy this gate. Preserve an existing run while repairing its monitor; do not relabel a preflight as training.

Record the run's task, dataset, seed, number of environments, budget, checkpoint, evaluation protocol, video and failure state. Keep licensed datasets and their derivative weights outside this repository until distribution rights have been checked.
