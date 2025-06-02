import ffmpeg, typer, os, sys, json, shutil
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<d>{time:YYYY-MM-DD ddd HH:mm:ss}</d> | <lvl>{level}</lvl> | <lvl>{message}</lvl>",
)
app = typer.Typer(pretty_exceptions_show_locals=False)


def parse_frame_name(fname: str):
    """return a tuple of frame_type and frame_index"""
    fn, fext = os.path.splitext(os.path.basename(fname))
    frame_type, frame_index = fn.split("_")
    return frame_type, int(frame_index)


def get_fps_ffmpeg(video_path: str):
    probe = ffmpeg.probe(video_path)
    # Find the first video stream
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"), None
    )
    if video_stream is None:
        raise ValueError("No video stream found")
    # Frame rate is given as a string fraction, e.g., '30000/1001'
    r_frame_rate = video_stream["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    return num / denom


@app.command()
def extract_keyframes_greedy(
    video_path: str,
    output_dir: str = None,
    threshold: float = 0.2,
    overwrite: bool = False,
):
    """
    run i-frames extractions and keyframes extraction and return a list of keyframe's paths
    """
    assert (
        threshold > 0
    ), f"threshold must be no negative, for i-frame extraction use extract-keyframes instead"

    iframes = extract_keyframes(
        video_path,
        output_dir=output_dir,
        threshold=0,
        overwrite=overwrite,
        append=False,
    )
    assert type(iframes) != type(None), f"i-frames extraction failed"
    kframes = extract_keyframes(
        video_path,
        output_dir=output_dir,
        threshold=threshold,
        overwrite=False,
        append=True,
    )
    assert type(kframes) != type(None), f"keyframes extraction failed"

    # remove kframes that are also iframes
    removed_kframes = []
    for fn in kframes:
        fname = os.path.basename(fn)
        if os.path.isfile(
            os.path.join(os.path.dirname(fn), fname.replace("kframe_", "iframe_"))
        ):
            os.remove(fn)
            removed_kframes.append(fn)
    if len(removed_kframes) > 0:
        logger.warning(f"removed {len(removed_kframes)} redundant kframes")
        kframes = [kf for kf in kframes if kf not in removed_kframes]

    frames = iframes + kframes
    logger.success(f"extracted {len(frames)} total frames")
    return frames


@app.command()
def extract_keyframes(
    video_path: str,
    output_dir: str = None,
    threshold: float = 0.3,
    overwrite: bool = False,
    append: bool = False,
):
    """extract keyframes as images into output_dir and return a list of keyframe's paths

    Args:
        output_dir: if not provided, will be in video_name/keyframes/
    """
    # Create output directory if it doesn't exist
    output_dir = output_dir if output_dir else os.path.dirname(video_path)
    vname, vext = os.path.splitext(os.path.basename(video_path))
    output_dir = os.path.join(output_dir, vname, "keyframes")
    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
            logger.warning(f"removed existing data: {output_dir}")
        elif not append:
            logger.error(f"overwrite is false and data already exists!")
            return None
    os.makedirs(output_dir, exist_ok=True)

    # Construct the ffmpeg-python pipeline
    stream = ffmpeg.input(video_path)
    config_dict = {
        "vsync": "0",
        "frame_pts": "true",
    }

    if threshold:
        # always add in the first frame by default
        filter_value = f"eq(n,0)+gt(scene,{threshold})"
        frame_name = "kframe"
        logger.info(f"Extracting Scene-changing frames with {filter_value}")
    else:
        filter_value = f"eq(pict_type,I)"
        # config_dict["skip_frame"] = "nokey"
        frame_name = "iframe"
        logger.info(f"Extracting I-Frames since no threshold provided: {filter_value}")

    stream = ffmpeg.filter(stream, "select", filter_value)
    stream = ffmpeg.output(stream, f"{output_dir}/{frame_name}_%d.jpg", **config_dict)

    # Execute the ffmpeg command
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        frames = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".jpg") and frame_name in f
        ]
        logger.success(f"{len(frames)} {frame_name} extracted to {output_dir}")
        return frames
    except ffmpeg.Error as e:
        logger.error(f"Error executing FFmpeg command: {e.stderr.decode()}")
    return None


@app.command()
def extract_audio(video_path: str, output_dir: str = None, overwrite: bool = False):
    """extracting audio of a video file into m4a without re-encoding
    ref: https://www.baeldung.com/linux/ffmpeg-audio-from-video#1-extracting-audio-without-re-encoding
    """
    # Create output directory if it doesn't exist
    output_dir = output_dir if output_dir else os.path.dirname(video_path)
    vname, vext = os.path.splitext(os.path.basename(video_path))
    output_dir = os.path.join(output_dir, vname)
    output_fname = os.path.join(output_dir, vname + ".m4a")
    if os.path.isfile(output_fname):
        if overwrite:
            os.remove(output_fname)
            logger.warning(f"removed existing data: {output_fname}")
        else:
            logger.error(f"overwrite is false and data already exists!")
            return None
    os.makedirs(output_dir, exist_ok=True)

    # Construct the ffmpeg-python pipeline
    stream = ffmpeg.input(video_path)
    config_dict = {"map": "0:a", "acodec": "copy"}
    stream = ffmpeg.output(stream, output_fname, **config_dict)

    # Execute the ffmpeg command
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        logger.success(f"audio extracted to {output_fname}")
        return output_fname
    except ffmpeg.Error as e:
        logger.error(f"Error executing FFmpeg command: {e.stderr.decode()}")
    return None


@app.command()
def extract_frames(
    video_path: str,
    output_dir: str = None,
    fps: int = None,
    every_x: int = None,
    overwrite: bool = False,
    append: bool = False,
    im_name_pattern: str = "frame_%05d.jpg",
):
    """extract frames as images into output_dir and return the list of frames' paths

    Args:
        output_dir: if not provided, will be in video_name/keyframes/
    """
    # Create output directory if it doesn't exist
    vname, vext = os.path.splitext(os.path.basename(video_path))
    output_dir = output_dir if output_dir else os.path.dirname(video_path)
    output_dir = os.path.join(output_dir, vname, "keyframes")
    if os.path.isdir(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
            logger.warning(f"removed existing data: {output_dir}")
        elif not append:
            logger.error(f"overwrite is false and data already exists in {output_dir}!")
            return None
    os.makedirs(output_dir, exist_ok=True)

    # Construct the ffmpeg-python pipeline
    stream = ffmpeg.input(video_path)
    config_dict = {
        "vsync": 0,  # preserves the original timestamps
        "frame_pts": 1,  # set output file's %d to the frame's PTS
    }
    if fps:
        # check FPS
        vid_fps = get_fps_ffmpeg(video_path)
        fps = min(vid_fps, fps)
        logger.info(f"{vname}{vext} FPS: {vid_fps}, extraction FPS: {fps}")
        config_dict["vf"] = f"fps={fps}"
    elif every_x:
        config_dict["vf"] = f"select=not(mod(n\,{every_x}))"

    logger.info(
        f"Extracting Frames into {output_dir} with these configs: \n{config_dict}"
    )
    stream = ffmpeg.output(stream, f"{output_dir}/{im_name_pattern}", **config_dict)

    # Execute the ffmpeg command
    try:
        ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        frames = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.endswith(".jpg")
        ]
        logger.success(f"{len(frames)} frames extracted to {output_dir}")
        return frames
    except ffmpeg.Error as e:
        logger.error(f"Error executing FFmpeg command: {e.stderr.decode()}")
    return None


if __name__ == "__main__":
    app()
