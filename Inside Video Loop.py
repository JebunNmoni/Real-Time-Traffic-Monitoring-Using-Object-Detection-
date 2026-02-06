current_time = time.time()
fps = 1 / (current_time - prev_time)
prev_time = current_time

vehicle_count = len(detections.tracker_id)

draw_metrics(frame, fps, vehicle_count)
