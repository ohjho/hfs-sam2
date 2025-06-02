import numpy as np
from tqdm import tqdm
import cv2, imageio, ffmpeg, os, time, shutil

def VidInfo(vid_path):
	'''
	returns a dictonary of 'duration', 'fps', 'frame_count', 'frame_height', 'frame_width',
							'format', 'fourcc'
	'''
	vcap = cv2.VideoCapture(vid_path)
	if not vcap.isOpened():
		# cannot read video
		if vid_path.startswith('https://'):
			# likely a ffmpeg without open-ssl support issue
			# https://github.com/opencv/opencv-python/issues/204
			return VidInfo(vid_path.replace('https://','http://'))
		else:
			return None

	info_dict = {
		'fps' : round(vcap.get(cv2.CAP_PROP_FPS),2), #int(vcap.get(cv2.CAP_PROP_FPS)),
		'frame_count': int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)), # number of frames should integars
		'duration': round(
			int(vcap.get(cv2.CAP_PROP_FRAME_COUNT)) / vcap.get(cv2.CAP_PROP_FPS),
			2), # round number of seconds to 2 decimals
		'frame_height': vcap.get(cv2.CAP_PROP_FRAME_HEIGHT),
		'frame_width': vcap.get(cv2.CAP_PROP_FRAME_WIDTH),
		'format': vcap.get(cv2.CAP_PROP_FORMAT),
		'fourcc': vcap.get(cv2.CAP_PROP_FOURCC)
	}
	vcap.release()
	return info_dict

def VidReader(vid_path, verbose = False, use_imageio = True):
	'''
	given a video file path, returns a list of images
	Args:
		vid_path: a MP4 file path
		use_imageio: if true, function returns a ImageIO reader object (RGB);
					otherwise, a list of CV2 array will be returned
	'''

	if use_imageio:
		vid = imageio.get_reader(vid_path, 'ffmpeg')
		return vid

	vcap = cv2.VideoCapture(vid_path)
	s_time = time.time()

	# try to determine the total number of frames in Vid
	frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_rate = int(vcap.get(cv2.CAP_PROP_FPS))
	if verbose:
		print(f'\t{frame_count} total frames in video {vid_path}')
		print(f'\t\t FPS: {frame_rate}')
		print(f'\t\t Video Duration: {frame_count/ frame_rate}s')

	# loop over frames
	results = []
	for i in tqdm(range(frame_count)):
		grabbed, frame = vcap.read()
		if grabbed:
			results.append(frame)

	# Output
	r_time = "{:.2f}".format(time.time() - s_time)
	if verbose:
		print(f'\t{vid_path} loaded in {r_time} ({frame_count/float(r_time)} fps)')
	vcap.release()
	return results

def get_vid_frame(n, vid_path):
	'''
	return frame(s) in np.array specified by i
	Args:
		n: list of int
	'''
	vreader = VidReader(vid_path, verbose = False, use_imageio = True)
	fcount = VidInfo(vid_path)['frame_count']

	if type(n) == list:
		return [vreader.get_data(i) if i in range(fcount) else None for i in n]
	elif type(n) == int:
		return vreader.get_data(n) if n in range(fcount) else None
	else:
		raise ValueError(f'n must be either int or list, {type(n)} detected.')

def vid_slicer(vid_path, output_path, start_frame, end_frame, keep_audio = False, overwrite = False):
	'''
	ref https://github.com/kkroening/ffmpeg-python/issues/184#issuecomment-493847192
	'''
	if not( os.path.isdir(os.path.dirname(output_path))):
		raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

	if os.path.isfile(output_path) and not overwrite:
		warnings.warn(f'{output_path} already exists but overwrite switch is False, nothing done.')
		return None

	input_vid = ffmpeg.input(vid_path)
	vid_info = VidInfo(vid_path)
	end_frame += 1

	if keep_audio:
		vid = (
			input_vid
			.trim(start_frame = start_frame, end_frame = end_frame)
			.setpts('PTS-STARTPTS')
		)
		aud = (
			input_vid
			.filter_('atrim', start = start_frame / vid_info['fps'], end = end_frame / vid_info['fps'])
			.filter_('asetpts', 'PTS-STARTPTS')
		)
		joined = ffmpeg.concat(vid, aud, v = 1, a =1).node
		output = ffmpeg.output(joined[0], joined[1], f'{output_path}').overwrite_output()
		output.run()
	else:
		(
			input_vid
			.trim   (start_frame = start_frame, end_frame = end_frame )
			.setpts ('PTS-STARTPTS')
			.output (f'{output_path}')
			.overwrite_output()
			.run()
		)
	return output_path

def vid_resize(vid_path, output_path, width, overwrite = False):
	'''
	use ffmpeg to resize the input video to the width given, keeping aspect ratio
	'''
	if not( os.path.isdir(os.path.dirname(output_path))):
		raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

	if os.path.isfile(output_path) and not overwrite:
		warnings.warn(f'{output_path} already exists but overwrite switch is False, nothing done.')
		return None

	input_vid = ffmpeg.input(vid_path)
	vid = (
		input_vid
		.filter('scale', width, -1)
		.output(output_path)
		.overwrite_output()
		.run()
	)
	return output_path

def vid_reduce_framerate(vid_path, output_path, new_fps, overwrite = False):
	'''
	use ffmpeg to resize the input video to the width given, keeping aspect ratio
	'''
	if not( os.path.isdir(os.path.dirname(output_path))):
		raise ValueError(f'output_path directory does not exists: {os.path.dirname(output_path)}')

	if os.path.isfile(output_path) and not overwrite:
		warnings.warn(f'{output_path} already exists but overwrite switch is False, nothing done.')
		return None

	input_vid = ffmpeg.input(vid_path)
	vid = (
		input_vid
		.filter('fps', fps = new_fps, round = 'up')
		.output(output_path)
		.overwrite_output()
		.run()
	)
	return output_path

def seek_frame_count(VidReader, cv2_frame_count, guess_within = 0.1,
	seek_rate = 1, bDebug = False):
	'''
	imageio/ffmpeg frame count could be different than cv2. this function
	returns the true frame count in the given vid reader. Returns None if frame
	count can't be determined
	Args:
		VidReader: ImageIO video reader object with method .get_data()
		cv2_frame_count: frame count from cv2
		guess_within: look for actual frame count within X% of cv2_frame_count
	'''
	max_guess = int(cv2_frame_count * (1-guess_within))
	seek_rate = max(seek_rate, 1)
	pbar = reversed(range(max_guess, cv2_frame_count, seek_rate))
	if bDebug:
		pbar = tqdm(pbar, desc = f'seeking frame')
		print(f'seeking from {max_guess} to {cv2_frame_count} with seek_rate of {seek_rate}')

	for i in pbar:
		try:
			im = VidReader.get_data(i)
		except IndexError:
			if bDebug:
				print(f'{i} not found.')
			continue
		# Frame Found
		if i+1 == cv2_frame_count:
			print(f'seek_frame_count: found frame count at {i+1}')
			return i + 1
		else:
			return seek_frame_count(VidReader, cv2_frame_count = i + seek_rate,
				guess_within= seek_rate / (i + seek_rate),
				seek_rate= int(seek_rate/2),
				bDebug = bDebug)
	return None

def VidWriter(lFrames, output_path, strFourcc = 'MP4V', verbose = False, intFPS = 20, crf = None,
				use_imageio = False):
	'''
	Given a list of images in numpy array format, it outputs a MP4 file
	Args:
		lFrames: list of numpy arrays or filename
		output_path: a MP4 file path
		strFourcc: four letter video codec; XVID is more preferable. MJPG results in high size video. X264 gives very small size video; see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
		crf: Constant Rate Factor for ffmpeg video compression
	'''
	s_time = time.time()

	if not output_path.endswith('.mp4'):
		raise ValueError(f'VidWriter: only mp4 video output supported.')

	if crf:
		crf = int(crf)
		if crf > 24 or crf < 18:
			raise ValueError(f'VidWriter: crf must be between 18 and 24')

	if not os.path.exists(os.path.dirname(output_path)):
		output_dir = os.path.dirname(output_path)
		print(f'\t{output_dir} does not exist.\n\tCreating video file output directory: {output_dir}')
		os.makedirs(output_dir)

	if use_imageio:
		writer = imageio.get_writer(output_path, fps = intFPS)
		for frame in tqdm(lFrames, desc = "Writing video using ImageIO"):
			if not type(frame) == np.ndarray:
				# read from filename
				if not os.path.isfile(frame):
					raise ValueError(f'VidWriter: lFrames must be list of images (np.array) or filenames')
				frame = imageio.imread(frame)

			writer.append_data(frame)
		writer.close()
	else:
		#init OpenCV Vid Writer:
		H , W = lFrames[0].shape[:2]
		#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
		fourcc = cv2.VideoWriter_fourcc(*strFourcc)
		if verbose:
			print(f'\tEncoding using fourcc: {strFourcc}')
		writer = cv2.VideoWriter(output_path, fourcc, fps = intFPS, frameSize = (W, H), isColor = True)

		for frame in tqdm(lFrames, desc = "Writing video using OpenCV"):
			writer.write(frame)
		writer.release()

	# Output
	r_time = "{:.2f}".format( max(time.time() - s_time, 0.01))
	if verbose:
		print(f'\t{output_path} written in {r_time} ({len(lFrames)/float(r_time)} fps)')

	if crf:
		if verbose:
			print(f'\tCompressing {output_path} with FFmpeg using crf: {crf}')

		isCompressed = VidCompress(output_path, crf = crf, use_ffmpy = False)

		if verbose:
			print(f'\tCompressed: {isCompressed}')

	return output_path

def im_dir_to_video(im_dir, output_path, fps, tup_im_extension = ('.jpg'),
		max_long_edge = 600, filename_len = 6, pixel_format = 'yuv420p',
		tqdm_func = tqdm):
	'''turn a directory of images into video using ffmpeg
		ref: https://github.com/kkroening/ffmpeg-python/issues/95#issuecomment-401428324
	Args:
		pixel_format: for list of supported formats see https://en.wikipedia.org/wiki/FFmpeg#Pixel_formats
		filename_len: ensure frame number are zero padded; 0 will skip this step
	'''
	if filename_len:
		# Ensure Filenames are Zero padded
		l_im_fp = [f for f in os.listdir(im_dir) if f.endswith(tup_im_extension)]
		l_im_fp = sorted(l_im_fp, key = lambda f: int(f.split('.')[0]))
		for f in tqdm_func(l_im_fp, desc = 'ensuring image filenames are zero padded'):
			fname, fext = os.path.splitext(f)
			padded_f = fname.zfill(filename_len) + fext
			if not os.path.isfile(os.path.join(im_dir,padded_f)):
				shutil.move(os.path.join(im_dir, f), os.path.join(im_dir, padded_f))
				# removed symlink to f as it will duplicate the frames in video generation
				# os.symlink(src = os.path.join(im_dir, padded_f), dst = os.path.join(im_dir, f))
			#TODO: ensure image size are divisible by 2

	im_dir += '' if im_dir.endswith('/') else '/'
	im_stream_string = f'{im_dir}*.jpg'
	# we need to escape special characters
	im_stream_string = im_stream_string.translate(
							str.maketrans(
								{'[': r'\[',
								']': r'\]'})
						)
	r = (
		ffmpeg
		.input(im_stream_string, pattern_type = 'glob', framerate=fps)
		.filter('format', pixel_format)
		# .filter('pad', 'ceil(iw/2)*2:ceil(ih/2)*2')
		.output(output_path)
		.run()
	)
	return output_path
#
# def VidCompress(input_path, output_path = None, crf = 24, use_ffmpy = False):
# 	'''
# 	Compress input_path video (mp4 only) using ffmpy
# 	crf: Constant Rate Factor for ffmpeg video compression, must be between 18 and 24
# 	use_ffmpy: use ffmpy instead of commandline call to ffmpeg
# 	'''
# 	if not input_path.endswith('.mp4'):
# 		print(f'\tFATAL: only mp4 videos supported.')
# 		return None
#
# 	output_fname = output_path if output_path else input_path
# 	tmp_fname = input_path.replace(".mp4","_tmp.mp4")
# 	os.rename(input_path, tmp_fname)
#
# 	try:
# 		if not use_ffmpy:
# 			#os.popen(f'ffmpeg -i {tmp_fname} -vcodec libx264 -crf {crf} {output_fname}')
#
# 			cmdOut = subprocess.Popen(['ffmpeg', '-i', tmp_fname, '-vcodec', 'libx264', '-crf', str(crf), output_fname],
# 										stdout = subprocess.PIPE,
# 										stderr = subprocess.STDOUT)
# 			stdout, stderr = cmdOut.communicate()
# 			if not stderr:
# 				os.remove(tmp_fname)
# 				return True
# 			else:
# 				return False
# 		else:
# 			ff = FFmpeg(
# 					inputs = {tmp_fname : None},
# 					outputs = {output_fname : f'-vcodec libx264 -crf {crf}'}
# 					)
# 			ff.run()
#
# 			os.remove(tmp_fname)
# 			return True
#
# 	except OSError as e:
# 		print(f'\tWARNING: Compression Failed; OSError\n\tLikely out of RAM\n\tError Msg: {e}')
# 		os.rename(tmp_fname, output_fname)
# 		return False
