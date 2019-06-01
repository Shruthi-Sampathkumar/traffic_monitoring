import os
import scipy.io as sci
from xml.dom import minidom
import xml.etree.ElementTree as ET
import tensorflow as tf
from dataset_utils import int64_feature, float_feature, bytes_feature, \
							write_auxiliary_file, read_auxiliary_file, \
							write_pbtxt_file




#xml_files = os.listdir(annotation_file)
#for file in xml_files:
	#if file.startswith('._'):
		#file = file.replace("._", "")

		#print(file)
		#root = ET.parse(annotation_file+'/'+file).getroot()

				

'''
def image_to_tfrecord(video_folder, annotation_file, tfrecord_filename):
	"""Convert Image to TFRecords.
		crop_folder_name = crop_names.txt
		crops_to_label = name : label
		Other than image name, label_text, tfrecord constains
		additional information include image, image shape, label_id. image and
		image shape are computed by reading image from disk, label_id is
		obtai pos_label=ned by a map "label_to_id" from label_text to label_id.
	"""
	# read the label file
	#filename_to_labels_dict = read_label_file(crops_to_label)
	images_list = os.listdir(video)
	# define image decoding graph
	inputs = tf.placeholder(dtype=tf.string)
	decoded_jpg = tf.image.decode_jpeg(inputs)
	# open tfRecord reader
	with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
		# open a session for image decoding
		with tf.Session() as sess:
			root = ET.parse(annotation_file).getroot()
			

			for item in root.findall("frame"):
				target_id = []
				frame_id = item.attrib['num']
				box_dict = {}
				for target in item.findall('target_list/target'):
					target_id.append(target.attrib['id'])
				for item1 in item.findall("target_list/target/box"):
					if 'xmin' not in box_dict.keys():
						box_dict['xmin'] = []
					box_dict['xmin'].append(float(item1.attrib['left']))
					if 'ymin' not in box_dict.keys():
						box_dict['ymin'] = []
					box_dict['ymin'].append(float(item1.attrib['top']))
					if 'xmax' not in box_dict.keys():
						box_dict['xmax'] = []
					box_dict['xmax'].append(float(item1.attrib['left']) + float(item1.attrib['width']))
					if 'ymax' not in box_dict.keys():
						box_dict['ymax'] = []
					box_dict['ymax'].append(float(item1.attrib['top']) + float(item1.attrib['height']))

				label = int(1) 
				source_id = frame_id
				tmp = frame_id.zfill(5)
				tmp = 'img' + tmp + '.jpg'
				print ('converting %s' % tmp)
				# read image
				image_data = tf.gfile.FastGFile(image_folder + tmp, 'rb').read()
				# decode image
				if tmp.endswith(('jpg','JPG')):
					image_data_decoded = sess.run(decoded_jpg,
													feed_dict={inputs:image_data})
					image_format = b'JPG'
				else:
					raise ValueError("image%s is not supported"%tmp)
				shape = list(image_data_decoded.shape)
				# create tf example
				example = tf.train.Example(features=tf.train.Features(feature={
					'image/format': bytes_feature(image_format),
					'image/encoded': bytes_feature(image_data),
					'image/filename' : bytes_feature(bytes(tmp, 'utf-8')),
					#'image/key/sha256' : bytes_feature(shape),
					'image/source_id' : bytes_feature(bytes(source_id, 'utf-8')),
					'image/height' : int64_feature(shape[0]),
					'image/width': int64_feature(shape[1]),
					'bbox/xmin' : float_feature(box_dict['xmin']),
					'bbox/xmax' : float_feature(box_dict['xmax']),
					'bbox/ymin' : float_feature(box_dict['ymin']),
					'bbox/ymax' : float_feature(box_dict['ymax']),
					'bbox/label/index' : int64_feature(label)
				}))
				# write example
				tfrecord_writer.write(example.SerializeToString()) 
				print("FRAME FOUND")  
'''	

def image_to_tfrecord(video_folder, annotation_file, tfrecord_filename):
	"""Convert Image to TFRecords.
		crop_folder_name = crop_names.txt
		crops_to_label = name : label
		Other than image name, label_text, tfrecord constains
		additional information include image, image shape, label_id. image and
		image shape are computed by reading image from disk, label_id is
		obtai pos_label=ned by a map "label_to_id" from label_text to label_id.
	"""
	# read the label file
	#filename_to_labels_dict = read_label_file(crops_to_label)
	video_image_list = os.listdir(video_folder)
	video_label_list = os.listdir(annotation_file)
	# define image decoding graph
	inputs = tf.placeholder(dtype=tf.string)
	decoded_jpg = tf.image.decode_jpeg(inputs)
	# open tfRecord reader
	with tf.python_io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
		# open a session for image decoding
		with tf.Session() as sess:
			for label_file in video_label_list:
				if label_file.startswith("._"):
					continue
				print("VIDEO--------------------------- ", label_file)
				frame = ""
				label_path = annotation_file + "/" + label_file
				video_name = label_file.strip("\n").replace(".txt","")
				with open(label_path, "r") as ff:
					annotations = ff.readlines()
					for line in annotations:
						line = line.strip()
						row = line.split(" ")
						if frame == row[0]:
							print("Appending")
							if row[2] == "Car" or row[2] == "Van" or row[2] == "Truck":
								label_class.append(int(1))
							elif row[2] == "Pedestrian" or row[2] == "Person_sitting" or row[2] == "Cyclist":
								label_class.append(int(2))
							else:
								label_class.append(int(0))
							xmin.append(float(row[6]))
							ymin.append(float(row[7]))
							xmax.append(float(row[8]))
							ymax.append(float(row[9]))

						else:
							if frame != "" and os.path.exists(image):

								print ('converting %s' % frame_name)
								# read image
								image_data = tf.gfile.FastGFile(image, 'rb').read()
								# decode image
								if frame_name.endswith(('png','PNG')):
									image_data_decoded = sess.run(decoded_jpg,
														feed_dict={inputs:image_data})
									image_format = b'PNG'
								else:
									raise ValueError("image%s is not supported"%frame_name)
								shape = list(image_data_decoded.shape)
								# create tf example
								example = tf.train.Example(features=tf.train.Features(feature={
									'image/format': bytes_feature(image_format),
									'image/encoded': bytes_feature(image_data),
									'image/filename' : bytes_feature(bytes(frame_name, 'utf-8')),
									#'image/key/sha256' : bytes_feature(shape),
									'image/source_id' : bytes_feature(bytes(source_id, 'utf-8')),
									'image/height' : int64_feature(shape[0]),
									'image/width': int64_feature(shape[1]),
									'bbox/xmin' : float_feature(xmin),
									'bbox/xmax' : float_feature(xmax),
									'bbox/ymin' : float_feature(ymin),
									'bbox/ymax' : float_feature(ymax),
									'bbox/label/index' : int64_feature(label_class)
								}))
								# write example
								tfrecord_writer.write(example.SerializeToString()) 
								print("FRAME FOUND") 

								xmin = []
								ymin = []
								xmax = []
								ymax = []
								label_class = []
								frame = row[0]
								source_id = row[0]
								frame_name = row[0].zfill(6) + ".png"
								image = video_folder + "/" + video_name + "/" + frame_name
								if os.path.exists(image):
									if row[2] == "Car" or row[2] == "Van" or row[2] == "Truck":
										label_class.append(int(1))
									elif row[2] == "Pedestrian" or row[2] == "Person_sitting" or row[2] == "Cyclist":
										label_class.append(int(2))
									else:
										label_class.append(int(0))

									xmin.append(float(row[6]))
									ymin.append(float(row[7]))
									xmax.append(float(row[8]))
									ymax.append(float(row[9]))	



							else:
								print("START FRAME")
								xmin = []
								ymin = []
								xmax = []
								ymax = []
								label_class = []
								frame = row[0]
								source_id = row[0]
								frame_name = row[0].zfill(6) + ".png"
								image = video_folder + "/" + video_name + "/" + frame_name
								if os.path.exists(image):
									if row[2] == "Car" or row[2] == "Van" or row[2] == "Truck":
										label_class.append(int(1))
									elif row[2] == "Pedestrian" or row[2] == "Person_sitting" or row[2] == "Cyclist":
										label_class.append(int(2))
									else:
										label_class.append(int(0))

									xmin.append(float(row[6]))
									ymin.append(float(row[7]))
									xmax.append(float(row[8]))
									ymax.append(float(row[9]))

						
def main():
	annotation_file = '/Volumes/My Passport/amir/lstm_data/KITTI Tracking dataset/training/label_02'
	image_folder = '/Volumes/My Passport/amir/lstm_data/KITTI Tracking dataset/training-1/image_02'
	tfrecord_filename = '/Volumes/My Passport/amir/lstm_data/KITTI Tracking dataset/kitti_car_pedes.tfrecord'
	image_to_tfrecord(image_folder, annotation_file , tfrecord_filename)

if __name__ == '__main__':
	#test()
	main()