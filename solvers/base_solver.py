class SIMULATOR:
    def __init__(self, verbose, cfg):
        # Additional initialization can be added in the derived classes
        self.verbose = verbose
        self.cfg = cfg

        # Initialize time tracking
        self.current_time = 0.0
        self.record_dt = cfg.record_dt  # Time interval between recordings
        self.next_record_time = 0.0  # Time of next recording
        self.end_time = cfg.record_dt * cfg.end_frame  # End time for simulation

        # Track frame numbers for output files
        self.record_frame = 0
        self.num_steps = 0

    def pre_process(self):
        # TODO implement in override
        pass

    def cal_dt(self):
        """Calculate and return the base timestep (without considering recording)"""
        # TODO implement in override
        pass

    def call_back(self):
        """Called after each recording"""
        # TODO implement in override
        pass

    def post_process(self):
        # TODO implement in override
        pass

    def dump(self):
        """Save simulation state at current_time"""
        # TODO implement in override
        pass

    def adjust_dt_for_recording(self, dt):
        """
        Adjust timestep to align with recording times.
        Returns adjusted timestep and whether we should record after this step.
        """
        time_remaining = self.next_record_time - self.current_time

        # If we've passed the recording time (shouldn't normally happen)
        if time_remaining <= 1e-10:  # Small tolerance for floating point comparison
            if self.verbose:
                print(
                    f"Warning: Current time {self.current_time:.6f} has passed the next recording time {self.next_record_time:.6f}."
                )
            return time_remaining, True

        # Calculate how many base timesteps remain until recording
        steps_to_record = time_remaining / dt

        if steps_to_record >= 2:
            # No adjustment needed
            return dt, False
        elif steps_to_record > 1:
            # Halve the timestep to get closer to recording time
            return dt / 2, False
        else:
            # Adjust timestep to exactly reach recording time
            return time_remaining, True

    def step(self, dt):
        """
        Perform a simulation step.
        """
        # TODO implement in override
        pass

    def early_stop(self):
        """
        Check if the simulation should stop early.
        Returns True if the simulation should stop, False otherwise.
        """
        # TODO implement in override
        return False

    def run(self):
        self.pre_process()

        # Initial recording at time 0
        self.dump()
        self.call_back()
        self.record_frame += 1
        self.next_record_time = min(self.current_time + self.record_dt, self.end_time)

        while self.current_time < self.end_time - 1e-10:
            # Calculate base timestep
            base_dt = self.cal_dt()

            # Adjust timestep for recording if needed
            dt, should_record = self.adjust_dt_for_recording(base_dt)

            if self.verbose:
                print(f"Time: {self.current_time:.6f}, dt: {dt:.6e}")

            # Perform simulation step
            self.step(dt)
            self.current_time += dt
            self.num_steps += 1

            # Handle recording if we've reached a recording time
            if should_record:
                if self.verbose:
                    print(f"Recording at time {self.current_time:.6f} (frame {self.record_frame})")
                    print(f"Number of steps: {self.num_steps}")
                self.dump()
                self.record_frame += 1
                self.next_record_time = min(self.current_time + self.record_dt, self.end_time)

            if self.early_stop():
                if self.verbose:
                    print(f"Early stopping at time {self.current_time:.6f}")
                break

        if self.verbose:
            print(f"Simulation completed at time {self.current_time:.6f}, total steps: {self.num_steps}")
        self.post_process()
