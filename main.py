import math
import random
import csv
import numpy as np
import cv2 as cv
import pymongo
from bson.son import SON


def is_in_circle(cx, cy, r, x, y):
    return math.sqrt((cx - x) ** 2 + (cy - y) ** 2) <= r


def find_stars(img):
    img = cv.medianBlur(img, 3)
    cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 0.2, 7,
                              param1=400, param2=12, minRadius=0, maxRadius=0)

    # circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 0.05, 5,
    #                           param1=350, param2=13, minRadius=0, maxRadius=0)

    # circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 0.01, 2,
    #                           param1=100, param2=11, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print((i[0], i[1]), " : ", i[2])
            cv.circle(cimg, (i[0], i[1]), i[2], (random.randint(0, 255), 0, random.randint(0, 255)), 2)
        cv.imshow('detected circles', cimg)
        cv.waitKey(0)
        cv.destroyAllWindows()


def calculate_hash():
    with open("hygdata_v3.csv", 'r') as f:
        dict_reader = csv.DictReader(f)
        stars_list = list(dict_reader)

    try:
        db_client = pymongo.MongoClient("mongodb://localhost:27017")
        current_db = db_client["stars_catalogue"]
        collection = current_db["stars4"]

        for row in stars_list:
            ra = float(row.get('ra'))
            dec = float(row.get('dec'))
            insert_value = {
                '_id': row.get('id'),
                'location': {'type': 'Point', 'coordinates': [ra, dec]},
                'mag': row.get('mag')
            }
            collection.insert_one(insert_value)

        collection.create_index([('location', '2dsphere')])

        for curr_star in collection.find():
            ra = (curr_star.get('location')).get('coordinates')[0]
            dec = (curr_star.get('location')).get('coordinates')[1]

            query = {'location':
                {
                    '$near': SON([(
                        '$geometry', SON([
                            ('type', 'Point'),
                            ('coordinates', [ra, dec])])),
                        ('$maxDistance', 300000)
                    ])
                }
            }

            lst = [0]*10
            for doc in collection.find(query):
                if is_in_circle(ra, dec, 0.3, doc.get('location').get('coordinates')[0],
                                doc.get('location').get('coordinates')[1]):
                    lst[0] += 1
                elif is_in_circle(ra, dec, 0.6, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[1] += 1
                elif is_in_circle(ra, dec, 0.9, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[2] += 1
                elif is_in_circle(ra, dec, 1.2, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[3] += 1
                elif is_in_circle(ra, dec, 1.5, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[4] += 1
                elif is_in_circle(ra, dec, 1.8, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[5] += 1
                elif is_in_circle(ra, dec, 2.1, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[6] += 1
                elif is_in_circle(ra, dec, 2.4, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[7] += 1
                elif is_in_circle(ra, dec, 2.7, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[8] += 1
                elif is_in_circle(ra, dec, 3, doc.get('location').get('coordinates')[0],
                                  doc.get('location').get('coordinates')[1]):
                    lst[9] += 1

            lst[0] -= 1
            hash = ''
            for k in range(10):
                if len(str(lst[k])) == 3:
                    hash += str(lst[k])
                elif len(str(lst[k])) == 2:
                    hash = hash + '0' + str(lst[k])
                elif len(str(lst[k])) == 1:
                    hash = hash + '00' + str(lst[k])
            collection.update_one(
                {'_id': curr_star.get('_id')}, {'$set': {'hash': hash}}
            )

    except Exception as error:
        print('error', error)


img = cv.imread('stars.png', 0)
find_stars(img)

calculate_hash()
