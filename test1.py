import os
import scipy.io as sci
from xml.dom import minidom
import xml.etree.ElementTree as ET
import tensorflow as tf
from dataset_utils import int64_feature, int64_feature_list, float_feature, float_feature_list, bytes_feature, \
							int64_feature1, bytes_feature_list, write_auxiliary_file, read_auxiliary_file, \
							float_feature1, bytes_feature1, write_pbtxt_file



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
				count = 0
				if label_file.startswith("._"):
					continue
				print("VIDEO--------------------------- ", label_file)
				frame = ""
				label_path = annotation_file + "/" + label_file
				video_name = label_file.strip("\n").replace(".txt","")
				with open(label_path, "r") as ff:
					xmins = []
					ymins = []
					xmaxs = []
					ymaxs = []
					image_formats = []
					image_datas = []
					filenames = []
					source_ids = []
					heights = []
					widths = []
					label_classes = []

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
								count += 1
								# read image
								image_data_tmp = tf.gfile.FastGFile(image, 'rb').read()
								image_datas.append(image_data_tmp)
								# decode image
								if frame_name.endswith(('png','PNG')):
									image_data_decoded = sess.run(decoded_jpg,
														feed_dict={inputs:image_data_tmp})
									image_formats.append(b'PNG')
								else:
									raise ValueError("image%s is not supported"%frame_name)
								
								shape = list(image_data_decoded.shape)
								heights.append(shape[0])
								widths.append(shape[1])

								filenames.append(frame_name)
								source_ids.append(source_id)
								xmins.append(xmin)
								ymins.append(ymin)
								xmaxs.append(xmax)
								ymaxs.append(ymax)
								label_classes.append(label_class)


								if count > 20:
									# Non sequential features
									context = tf.train.Features(feature={
										'image/format': bytes_feature1(image_formats),
										'image/filename': bytes_feature1([x.encode('utf-8') for x in filenames]),
										'image/source_id':bytes_feature1([x.encode('utf-8') for x in source_ids]),
										'image/height':int64_feature1(heights),
										'image/width':int64_feature1(widths),
										})
									xmins1=[]
									for i in range(len(xmins)):
										#inner_list = []
										#for j in range(len(xmins[i])):
										xmins1.append(float_feature_list(xmins[i]))
										
										#xmins1.append(inner_list)	
									ymins1=[]			
									for i in range(len(ymins)):
										#inner_list = []
										#for j in range(len(ymins[i])):
										ymins1.append(float_feature_list(ymins[i]))
										#ymins1.append(inner_list)		
									xmaxs1=[]			
									for i in range(len(xmaxs)):
										#inner_list = []
										#for j in range(len(xmaxs[i])):
										xmaxs1.append(float_feature_list(xmaxs[i]))
										#xmaxs1.append(inner_list)	
									ymaxs1=[]			
									for i in range(len(ymaxs)):
										#inner_list = []
										#for j in range(len(ymaxs[i])):
										ymaxs1.append(float_feature_list(ymaxs[i]))
										#ymaxs1.append(inner_list)
									label_classes1 = []	
									for i in range(len(label_classes)):
										#inner_list = []
										#for j in range(len(label_classes[i])):
										label_classes1.append(int64_feature_list(label_classes[i]))
										#label_classes1.append(inner_list)	
									#print(label_classes)						
									
									#print(len(xmins[0]), len(xmins1[0]))
									#print(len(xmins),len(xmins))
									#print(xmins)
									#print(len(image_datas[1]))
									image_datas1=[]
									for i in range(len(image_datas)):
										image_datas1.append(bytes_feature_list(image_datas[i]))
										#image_datas1.append(tf.train.FeatureList(feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'image_datas[i]']))))
									#label_classes1 = tf.train.FeatureList(feature=label_classes1)
									#Sequential features
									tf_features={
										'image/encoded': image_datas1,
										'bbox/xmin': xmins1,
										'bbox/xmax': xmaxs1,
										'bbox/ymin': ymins1,
										'bbox/ymax': ymaxs1,
										'bbox/label/index': label_classes1
									}

									feature_lists = tf.train.FeatureLists(feature_list=(x for x in tf_features))
									# Make single sequence example
									tf_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
										
									# write example
									tfrecord_writer.write(tf_example.SerializeToString()) 
									print("FRAME FOUND") 
									break

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
	annotation_file = '/Volumes/ShruthiWD/amir/lstm_data1/KITTI Tracking dataset/training/label_02'
	image_folder = '/Volumes/ShruthiWD/amir/lstm_data1/KITTI Tracking dataset/training-1/image_02'
	tfrecord_filename = '/Volumes/ShruthiWD/amir/lstm_data1/KITTI Tracking dataset/kitti_car_pedes_20_1.tfrecord'
	image_to_tfrecord(image_folder, annotation_file , tfrecord_filename)

if __name__ == '__main__':
	#test()
	main()