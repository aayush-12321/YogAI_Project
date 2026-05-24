import os
import time
import cv2


def record_yoga_session_mp4():
    # Initialize webcam (0 is usually the default built-in camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    # Get video dimensions from your webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Fallback if FPS is reported incorrectly by hardware
    if fps <= 0 or fps > 60:
        fps = 30

    # CHANGED: Use 'mp4v' codec for native MP4 recording
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Temporary MP4 file name
    temp_filename = "temp_recording.mp4"
    out = cv2.VideoWriter(
        temp_filename, fourcc, fps, (frame_width, frame_height)
    )

    countdown_duration = 10  # seconds
    recording_duration = 60  # seconds

    start_time = time.time()
    state = "COUNTDOWN"  # States: COUNTDOWN, RECORDING, DONE

    print(
        f"Starting {countdown_duration}-second countdown. Press 'q' to cancel."
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Flip horizontally for a mirror effect (easier to position your body)
        frame = cv2.flip(frame, 1)
        elapsed = time.time() - start_time

        if state == "COUNTDOWN":
            remaining = int(countdown_duration - elapsed)

            # Display countdown and cancel cue on screen
            cv2.putText(
                frame,
                f"Starting in: {remaining}s",
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 165, 255),
                3,
            )
            cv2.putText(
                frame,
                "Press 'q' to cancel",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if elapsed >= countdown_duration:
                state = "RECORDING"
                start_time = time.time()  # Reset timer for the 1-minute recording
                print("Recording started!")

        elif state == "RECORDING":
            rec_elapsed = time.time() - start_time
            remaining_rec = int(recording_duration - rec_elapsed)

            # Write frame to the MP4 file
            out.write(frame)

            # Display recording status metrics
            cv2.putText(
                frame,
                "• RECORDING",
                (50, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Time Left: {remaining_rec}s",
                (50, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' to abort",
                (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )

            if rec_elapsed >= recording_duration:
                state = "DONE"
                break

        # Display preview
        cv2.imshow("YogAI Video Capture (MP4)", frame)

        # Middle-of-session cancellation: Press 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nRecording cancelled mid-session.")
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Post-processing file naming logic
    if state == "DONE":
        print("\nRecording complete!")
        user_filename = input(
            "Enter the desired filename (without extension): "
        ).strip()

        if not user_filename:
            user_filename = f"yoga_session_{int(time.time())}"

        # CHANGED: Append .mp4 extension
        final_filename = f"{user_filename}.mp4"

        try:
            os.rename(temp_filename, final_filename)
            print(f"Success! Video saved as: {os.path.abspath(final_filename)}")
        except Exception as e:
            print(f"Error saving file: {e}. Saved as {temp_filename} instead.")
    else:
        # Automatically deletes the temporary fragmented MP4 file if you press 'q'
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print("Incomplete video file discarded cleanly.")


if __name__ == "__main__":
    record_yoga_session_mp4()