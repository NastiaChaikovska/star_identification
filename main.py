import csv
import math
import cv2 as cv
import pymongo
from bson.son import SON

db_client = pymongo.MongoClient("mongodb://localhost:27017")
current_db = db_client["stars_catalogue"]
collection = current_db["stars5"]

NUM_OF_CIRCLES = 10


def is_in_circle(current_star, r, star):
    return (math.dist(current_star, star)) ** 2 <= (r ** 2)


def calculate_hash_for_star_photo(star_coord, stars, deg_in_one_pixel, cimg):
    lst_hash = [0] * NUM_OF_CIRCLES
    step = 0.3 / deg_in_one_pixel

    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step), (125, 205, 5), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 1 * step), (175, 5, 55), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 2 * step), (5, 5, 105), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 3 * step), (115, 70, 205), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 4 * step), (255, 175, 17), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 5 * step), (105, 175, 170), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 6 * step), (5, 175, 17), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 7 * step), (215, 75, 215), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 8 * step), (175, 115, 100), 2)
    cv.circle(cimg, (star_coord[0], star_coord[1]), int(step + 9 * step), (75, 75, 125), 2)

    for star in stars:
        for i in range(NUM_OF_CIRCLES):
            if (star_coord[0] - (step + i * step) <= star[0] <= star_coord[0] + (step + i * step)) \
                    and (star_coord[1] - (step + i * step) <= star[1] <= star_coord[1] + (step + i * step)):
                if is_in_circle(star_coord, step + i * step, (star[0], star[1])):
                    lst_hash[i] += 1
                    if i == 0:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (125, 205, 5), 3)
                    elif i == 1:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (175, 5, 55), 3)
                    elif i == 2:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (5, 5, 105), 3)
                    elif i == 3:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (115, 70, 205), 3)
                    elif i == 4:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (255, 175, 17), 5)
                    elif i == 5:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (105, 175, 170), 5)
                    elif i == 6:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (5, 175, 17), 5)
                    elif i == 7:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (215, 75, 215), 3)
                    elif i == 8:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (175, 115, 100), 3)
                    elif i == 9:
                        cv.circle(cimg, (int(star[0]), int(star[1])), int(star[2]), (75, 75, 125), 3)
                    break

    cv.circle(cimg, (star_coord[0], star_coord[1]), 3, (5, 5, 255), 5)  # current star
    lst_hash[0] -= 1

    collection.create_index([('hash', 'text')])
    cv.imshow('res hash', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print('lst', lst_hash)
    identify_star(lst_hash)
    return lst_hash


def find_stars(img):
    field_of_view = 3.73  # 4.22   # 3.73
    dimention = img.shape
    stars = []

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    i = 0
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        stars.append([x+w/2, y+w/2, w/2])
        print(i, '. ', (x, y), ": ", w/2)
        i += 1
        if i == 97:
            jst_star = (x+int(w/2), y+int(w/2))
        if (dimention[0] / 2 - 150 <= y <= dimention[0] / 2 + 150) \
                and (dimention[1] / 2 - 250 <= x <= dimention[1] / 2 - 100):
            star1_coord = (x+int(w/2), y+int(w/2))
            print('\nstar1:', (x+int(w/2), y+int(w/2)), ": ", w/2)
            cv.circle(img, (x+int(w/2), y+int(w/2)), int(w/2), (255, 5, 5), 3)
        elif (dimention[0] / 2 - 150 <= y <= dimention[0] / 2 + 150) \
                and (dimention[1] / 2 + 100 <= x <= dimention[1] / 2 + 250):
            star2_coord = (x+int(w/2), y+int(w/2))
            print('\nstar2:', (x+int(w/2), y+int(w/2)), ": ", w/2)
            cv.circle(img, (x+int(w/2), y+int(w/2)), int(w/2), (5, 255, 5), 3)
        else:
            cv.circle(img, (x+int(w/2), y+int(w/2)), int(w/2), (5, 5, 255), 2)

    cv.imshow('Stars', img)
    cv.waitKey(0)

    deg_in_one_pixel = field_of_view / dimention[1]
    calculate_hash_for_star_photo(star1_coord, stars, deg_in_one_pixel, img)
    return stars


def calculate_hash_for_db():
    with open("hygdata_v3.csv", 'r') as f:
        dict_reader = csv.DictReader(f)
        stars_list = list(dict_reader)

    try:
        # for row in stars_list:
        #     ra = float(row.get('ra'))
        #     dec = float(row.get('dec'))
        #     insert_value = {
        #         '_id': row.get('id'),
        #         'location': {'type': 'Point', 'coordinates': [ra, dec]},
        #         'mag': row.get('mag')
        #     }
        #     collection.insert_one(insert_value)

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

            lst_hash = [0] * NUM_OF_CIRCLES
            step = 0.3
            for doc in collection.find(query):
                for i in range(NUM_OF_CIRCLES):
                    if is_in_circle(ra, dec, 0.3 + i * step, doc.get('location').get('coordinates')[0],
                                    doc.get('location').get('coordinates')[1]):
                        lst_hash[i] += 1
                        break

            lst_hash[0] -= 1
            hash = ''
            for k in range(NUM_OF_CIRCLES):
                if len(str(lst_hash[k])) == 3:
                    hash += str(lst_hash[k])
                elif len(str(lst_hash[k])) == 2:
                    hash = hash + '0' + str(lst_hash[k])
                elif len(str(lst_hash[k])) == 1:
                    hash = hash + '00' + str(lst_hash[k])
            print(hash)
            collection.update_one(
                {'_id': curr_star.get('_id')}, {'$set': {'hash': hash}}
            )

    except Exception as error:
        print('error', error)


def identify_star(lst_hash):
    print()
    lst_lt, lst_gt = [0] * NUM_OF_CIRCLES, [0] * NUM_OF_CIRCLES
    lt_hash, gt_hash = '', ''
    for k in range(NUM_OF_CIRCLES):
        lt = lst_hash[k] - 5
        gt = lst_hash[k] + 5
        lst_lt[k] = lt if lt > 0 else 0
        lst_gt[k] = gt

        if len(str(lt)) == 3:
            lt_hash += str(lt)
        elif len(str(lt)) == 2:
            if lt < 0:
                lt_hash += '000'
            else:
                lt_hash = lt_hash + '0' + str(lt)
        elif len(str(lt)) == 1:
            lt_hash = lt_hash + '00' + str(lt)

        if len(str(gt)) == 3:
            gt_hash += str(gt)
        elif len(str(gt)) == 2:
            gt_hash = gt_hash + '0' + str(gt)
        elif len(str(gt)) == 1:
            gt_hash = gt_hash + '00' + str(gt)

    print('lt_hash', lst_lt)
    print('gt_hash', lst_gt)

    for star in collection.find({'hash': {'$gt': lt_hash, '$lt': gt_hash}}):
        star_lst_hash = [0] * NUM_OF_CIRCLES

        for i in range(10):
            star_lst_hash[i] = int(star.get('hash')[(i * 3):3 + 3 * i])

        c = True
        for i in range(NUM_OF_CIRCLES):
            if lst_hash[i] - 5 < star_lst_hash[i] < lst_hash[i] + 5:
                continue
            else:
                c = False
                break

        if c:
            print('founded star: ', star)


img = cv.imread('img3736.png')
find_stars(img)

# calculate_hash_for_db()