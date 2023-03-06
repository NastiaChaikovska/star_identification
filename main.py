import csv
import math
import cv2 as cv
import pymongo
from bson.son import SON

db_client = pymongo.MongoClient("mongodb://localhost:27017")
current_db = db_client["stars_catalogue"]
collection = current_db["stars05"]

NUM_OF_CIRCLES = 10


def is_in_circle(current_star, r, star):
    return (math.dist(current_star, star)) ** 2 <= (r ** 2)


def calculate_hash_for_star_photo(star_coord, stars, deg_in_one_pixel, cimg):
    lst_hash = [0] * NUM_OF_CIRCLES
    step = 0.05 / deg_in_one_pixel

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

    num_of_stars = 0
    c = False
    for star in stars:
        for i in range(NUM_OF_CIRCLES):
            if (star_coord[0] - (step + i * step) <= star[0] <= star_coord[0] + (step + i * step)) \
                    and (star_coord[1] - (step + i * step) <= star[1] <= star_coord[1] + (step + i * step)):
                if is_in_circle(star_coord, step + i * step, (star[0], star[1])):
                    lst_hash[i] += 1
                    num_of_stars += 1
                    if (num_of_stars == 50):
                        c = True
                        break
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
            if c == True:
                break
    cv.circle(cimg, (star_coord[0], star_coord[1]), 3, (5, 5, 255), 2)  # current star
    lst_hash[0] -= 1

    collection.create_index([('hash', 'text')])
    cv.imwrite('star2_hash.png', cimg)

    cv.imshow('res hash', cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    print('lst', lst_hash)
    identify_star(lst_hash)
    return lst_hash


def find_stars(img):
    field_of_view = 3.3  # 4.22   # 3.73
    dimention = img.shape
    stars = []

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda x: cv.minEnclosingCircle(x)[1], reverse=True)

    i = 0
    for contour in sorted_contours:
        (x, y), radius_star = cv.minEnclosingCircle(contour)
        stars.append([x, y, radius_star])
        print(i, '. ', (x, y), ": ", radius_star)
        i += 1
        if (dimention[0] / 2 - 130 <= y <= dimention[0] / 2 - 75) \
                and (dimention[1] / 2 - 725 <= x <= dimention[1] / 2 - 525):
            star1_coord = (int(x), int(y))
            print('\nstar1:', (x, y), ": ", radius_star)
            cv.circle(img, (int(x), int(y)), int(radius_star), (255, 5, 5), 3)

        elif (dimention[0] / 2 <= y <= dimention[0] / 2 + 125) \
                and (dimention[1] / 2 -50 <= x <= dimention[1] / 2 +50):
            star2_coord = (int(x), int(y))
            print('\nstar2:', (x, y), ": ", radius_star)
            cv.circle(img, (int(x), int(y)), int(radius_star), (5, 255, 5), 3)

        elif (dimention[0] / 2 - 130 <= y <= dimention[0] / 2 - 75) \
                and (dimention[1] / 2 + 525 <= x <= dimention[1] / 2 + 725):
            star3_coord = (int(x), int(y))
            print('\nstar2:', (x, y), ": ", radius_star)
            cv.circle(img, (int(x), int(y)), int(radius_star), (5, 255, 5), 3)

        else:
            cv.circle(img, (int(x), int(y)), int(radius_star), (5, 5, 255), 2)

    cv.imshow('Stars', img)
    cv.waitKey(0)

    deg_in_one_pixel = field_of_view / dimention[1]
    calculate_hash_for_star_photo(star1_coord, stars, deg_in_one_pixel, img)
    calculate_hash_for_star_photo(star2_coord, stars, deg_in_one_pixel, img)
    calculate_hash_for_star_photo(star3_coord, stars, deg_in_one_pixel, img)
    return stars


def calculate_hash_for_db():
    with open("hygdata_v3.csv", 'r') as f:
        dict_reader = csv.DictReader(f)
        stars_list = list(dict_reader)

    try:
        for row in stars_list:
            ra = float(row.get('ra'))
            dec = float(row.get('dec'))
            mag = float(row.get('mag'))

            insert_value = {
                '_id': row.get('id'),
                'location': {'type': 'Point', 'coordinates': [ra, dec]},
                'mag': mag
            }
            collection.insert_one(insert_value)

        collection.create_index([('location', '2dsphere')])

        for curr_star in collection.find():
            ra = (curr_star.get('location')).get('coordinates')[0]
            dec = (curr_star.get('location')).get('coordinates')[1]
            sort_by_mag = [('mag', pymongo.ASCENDING)]
            query = {'location':
                {
                    '$near': SON([(
                        '$geometry', SON([
                            ('type', 'Point'),
                            ('coordinates', [ra, dec])])),
                        # ('$maxDistance', 100000)
                        ('$maxDistance', 50000)
                    ])
                }
            }

            lst_hash = [0] * NUM_OF_CIRCLES
            step = 0.05
            for doc in collection.find(query).sort(sort_by_mag).limit(51):
                for i in range(NUM_OF_CIRCLES):
                    if is_in_circle([ra, dec], step + i * step, [doc.get('location').get('coordinates')[0],
                                                                 doc.get('location').get('coordinates')[1]]):
                        lst_hash[i] += 1
                        break

            lst_hash[0] -= 1 if lst_hash[0] > 0 else 0
            hash = ''
            hash_sum = 0
            for k in range(NUM_OF_CIRCLES):
                hash_sum += lst_hash[k]
                if len(str(lst_hash[k])) == 2:
                    hash += str(lst_hash[k])
                elif len(str(lst_hash[k])) == 1:
                    hash = hash + '0' + str(lst_hash[k])
                elif len(str(lst_hash[k])) == 0:
                    hash = hash + '00' + str(lst_hash[k])
            print(hash_sum)
            collection.update_one(
                {'_id': curr_star.get('_id')}, {'$set': {'hash': hash}}
            )

    except Exception as error:
        print('error', error)


def identify_star(lst_hash):
    lst_lt, lst_gt = [0] * NUM_OF_CIRCLES, [0] * NUM_OF_CIRCLES
    lt_hash, gt_hash = '', ''
    for k in range(NUM_OF_CIRCLES):
        lt = lst_hash[k] - 3
        gt = lst_hash[k] + 3
        lst_lt[k] = lt if lt > 0 else 0
        lst_gt[k] = gt

        if len(str(lt)) == 2:
            lt_hash = lt_hash + str(lt)
        elif len(str(lt)) == 1:
            lt_hash = lt_hash + '0' + str(lt)

        if len(str(gt)) == 2:
            gt_hash += str(gt)
        elif len(str(gt)) == 1:
            gt_hash = gt_hash + '0' + str(gt)

    print('lt_hash', lst_lt)
    print('gt_hash', lst_gt)

    for star in collection.find({'hash': {'$gt': lt_hash, '$lt': gt_hash}}):
        star_lst_hash = [0] * NUM_OF_CIRCLES

        for i in range(NUM_OF_CIRCLES):
            star_lst_hash[i] = int(star.get('hash')[(i * 2):2 + 2 * i])

        c = True
        for i in range(NUM_OF_CIRCLES):
            if lst_lt[i] < star_lst_hash[i] < lst_gt[i]:
                continue
            else:
                c = False
                break

        if c:
            print('founded star: ', star)


# calculate_hash_for_db()

img = cv.imread('img3301832.png')
find_stars(img)
