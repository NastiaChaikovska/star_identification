import csv
import math
import cv2 as cv
import pymongo
import datetime
import numpy as np
import matplotlib.pyplot as plt
from bson.son import SON

db_client = pymongo.MongoClient("mongodb://localhost:27017")
current_db = db_client["stars_catalogue"]
collection = current_db["stars"]

DISTANCE_DEG = 3
NUM_OF_CIRCLES = 10
FIELD_OF_VIEW = 17.45


def is_in_circle(current_star, r, star):
    return (math.dist(current_star, star)) ** 2 <= (r ** 2)


def spherical_to_cartesian(l, phi, r):    # longitude  l, latitude phi
    x = r * math.cos(math.radians(phi)) * math.cos(math.radians(l))
    y = r * math.cos(math.radians(phi)) * math.sin(math.radians(l))
    z = r * math.sin(math.radians(phi))
    return x, y, z


def cartesian_to_spherical(cartesian_coordinates):
    x = cartesian_coordinates[2]
    phi = math.asin(x)
    y = cartesian_coordinates[1] / math.cos(phi)
    l = math.asin(y)
    return [phi * 180 / math.pi, l * 180 / math.pi]   # longitude  l, latitude phi


def import_data_from_csv_to_db():
    with open("hygdata_v3.csv", 'r') as f:
        dict_reader = csv.DictReader(f)
        stars_list = list(dict_reader)
    try:
        for row in stars_list:
            ra = float(row.get('ra')) * 15 - 180
            dec = float(row.get('dec'))
            mag = float(row.get('mag'))

            insert_value = {
                '_id': row.get('id'),
                'location': {'type': 'Point', 'coordinates': [ra, dec]},
                'mag': mag
            }
            collection.insert_one(insert_value)
    except Exception as error:
        print('error', error)


def calculate_hash_for_db():
    collection.create_index([('location', '2dsphere')])

    for curr_star in collection.find():
        ra = (curr_star.get('location')).get('coordinates')[0]
        dec = (curr_star.get('location')).get('coordinates')[1]
        sort_by_mag = [('mag', pymongo.ASCENDING)]
        query = {
            'location': {
                '$near': {
                    '$geometry': {
                        'type': 'Point',
                        'coordinates': [ra, dec]
                    },
                    '$maxDistance': DISTANCE_DEG * (2 * 3.14159 * 6378137) / 360
                }
            }
        }

        lst_hash = [0] * NUM_OF_CIRCLES
        step = DISTANCE_DEG / NUM_OF_CIRCLES
        for doc in collection.find(query).sort(sort_by_mag).limit(91):
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
        collection.update_one(
            {'_id': curr_star.get('_id')}, {'$set': {'hash': hash}}
        )


def generate_night_sky_photo(star_id):
    star = collection.find_one({'_id': star_id})
    star_ra = star.get('location').get('coordinates')[0]
    star_dec = star.get('location').get('coordinates')[1]
    star_mag = star.get('mag')
    print('\nСправжні координати спостерігача: [', star_ra, ', ', star_dec, ']\n', sep='')
    collection.create_index([('location', '2dsphere')])
    sort_by_mag = [('mag', pymongo.ASCENDING)]
    query = {'location':
        {
            '$near': SON([(
                '$geometry', SON([
                    ('type', 'Point'),
                    ('coordinates', [star_ra, star_dec])])),
                ('$maxDistance', 7 * (2 * 3.14159 * 6378137) / 360)
            ])
        }
    }

    ra, dec, mag = [], [], []
    for num, star in enumerate(collection.find(query).sort(sort_by_mag)):
        ra.append(star.get('location').get('coordinates')[0])
        dec.append(star.get('location').get('coordinates')[1])
        star_mag = abs(star.get('mag'))
        size = math.exp(-star_mag) + 0.1
        mag.append(15 * size)
    ra.append(star_ra)
    dec.append(star_dec)
    mag.append(25 * (1 / star_mag))
    colors = ['red' if i == len(ra) else 'blue' for i in range(len(ra))]
    figure, axes = plt.subplots(figsize=(10, 10))
    plt.scatter(ra, dec, s=mag, c=colors)
    plt.scatter(star_ra, star_dec, s=5 * (1 / star_mag), c='red')
    plt.axis('equal')
    axes.set_facecolor('black')
    plt.grid(True)
    plt.xlim(min(ra), max(ra))
    plt.axis('off')
    plt.savefig('night_sky_photo.png', facecolor='black', format='png', dpi=300, bbox_inches='tight')


def find_stars(img):
    stars = []
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda x: cv.minEnclosingCircle(x)[1], reverse=True)

    for contour in sorted_contours:
        (x, y), radius_star = cv.minEnclosingCircle(contour)
        cv.circle(img, (int(x), int(y)), int(radius_star), (5, 5, 255), 2)
        stars.append([x, y, radius_star])
    cv.imwrite('founded_stars.png', img)
    return stars


def select_stars_from_photo(stars, img):
    dimention = img.shape
    for star in stars:
        if (dimention[0] / 2 + 200 <= star[1] <= dimention[0] / 2 + 450) \
                and (dimention[1] / 2 - 700 <= star[0] <= dimention[1] / 2 - 600):
            star1_coord = (int(star[0]), int(star[1]))
        elif (dimention[0] / 2 - 650 <= star[1] <= dimention[0] / 2 - 425) \
                and (dimention[1] / 2 - 55 <= star[0] <= dimention[1] / 2 + 55):
            star2_coord = (int(star[0]), int(star[1]))
        elif (dimention[0] / 2 + 300 <= star[1] <= dimention[0] / 2 + 450) \
                and (dimention[1] / 2 + 500 <= star[0] <= dimention[1] / 2 + 650):
            star3_coord = (int(star[0]), int(star[1]))
        elif (dimention[0] / 2 - 220 <= star[1] <= dimention[0] / 2 - 160) \
                and (dimention[1] / 2 - 800 <= star[0] <= dimention[1] / 2 - 550):
            star4_coord = (int(star[0]), int(star[1]))
        elif (dimention[0] / 2 + 400 <= star[1] <= dimention[0] / 2 + 655) \
                and (dimention[1] / 2 - 45 <= star[0] <= dimention[1] / 2 + 45):
            star5_coord = (int(star[0]), int(star[1]))
        elif (dimention[0] / 2 - 30 <= star[1] <= dimention[0] / 2 + 30) \
                and (dimention[1] / 2 + 525 <= star[0] <= dimention[1] / 2 + 800):
            star6_coord = (int(star[0]), int(star[1]))

    star1_dist_to_center_photo = ((star1_coord[0] - dimention[1] / 2) ** 2 + (
            star1_coord[1] - dimention[0] / 2) ** 2) ** 0.5
    star2_dist_to_center_photo = ((star2_coord[0] - dimention[1] / 2) ** 2 + (
            star2_coord[1] - dimention[0] / 2) ** 2) ** 0.5
    star3_dist_to_center_photo = ((star3_coord[0] - dimention[1] / 2) ** 2 + (
            star3_coord[1] - dimention[0] / 2) ** 2) ** 0.5
    star4_dist_to_center_photo = ((star4_coord[0] - dimention[1] / 2) ** 2 + (
            star1_coord[1] - dimention[0] / 2) ** 2) ** 0.5
    star5_dist_to_center_photo = ((star5_coord[0] - dimention[1] / 2) ** 2 + (
            star2_coord[1] - dimention[0] / 2) ** 2) ** 0.5
    star6_dist_to_center_photo = ((star6_coord[0] - dimention[1] / 2) ** 2 + (
            star3_coord[1] - dimention[0] / 2) ** 2) ** 0.5

    triple_stars_1 = [{star1_coord: star1_dist_to_center_photo},
                      {star2_coord: star2_dist_to_center_photo},
                      {star3_coord: star3_dist_to_center_photo}]
    triple_stars_2 = [{star4_coord: star4_dist_to_center_photo},
                      {star5_coord: star5_dist_to_center_photo},
                      {star6_coord: star6_dist_to_center_photo}]

    triples_of_stars = [triple_stars_1, triple_stars_2]
    return triples_of_stars


def calculate_hash_for_star_photo(star_coord, stars, img, DEGREES_PER_PIXEL, image_name):
    lst_hash = [0] * NUM_OF_CIRCLES
    step = (DISTANCE_DEG / NUM_OF_CIRCLES) / DEGREES_PER_PIXEL

    cv.circle(img, (star_coord[0], star_coord[1]), int(step), (125, 205, 5), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 1 * step), (175, 5, 55), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 2 * step), (5, 5, 105), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 3 * step), (115, 70, 205), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 4 * step), (255, 175, 17), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 5 * step), (105, 175, 170), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 6 * step), (5, 175, 17), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 7 * step), (215, 75, 215), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 8 * step), (175, 115, 100), 2)
    cv.circle(img, (star_coord[0], star_coord[1]), int(step + 9 * step), (75, 75, 125), 2)

    num_of_stars = 0
    c = False
    for star in stars:
        for i in range(NUM_OF_CIRCLES):
            if (star_coord[0] - (step + i * step) <= star[0] <= star_coord[0] + (step + i * step)) \
                    and (star_coord[1] - (step + i * step) <= star[1] <= star_coord[1] + (step + i * step)):
                if is_in_circle(star_coord, step + i * step, (star[0], star[1])):
                    lst_hash[i] += 1
                    num_of_stars += 1
                    if num_of_stars == 90:
                        c = True
                        break
                    if i == 0:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (125, 205, 5), 2)
                    elif i == 1:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (175, 5, 55), 2)
                    elif i == 2:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (5, 5, 105), 2)
                    elif i == 3:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (115, 70, 205), 2)
                    elif i == 4:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (255, 175, 17), 2)
                    elif i == 5:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (105, 175, 170), 2)
                    elif i == 6:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (5, 175, 17), 2)
                    elif i == 7:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (215, 75, 215), 2)
                    elif i == 8:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (175, 115, 100), 2)
                    elif i == 9:
                        cv.circle(img, (int(star[0]), int(star[1])), int(star[2]), (75, 75, 125), 2)
                    break
            if c:
                break
    cv.circle(img, (star_coord[0], star_coord[1]), 3, (5, 5, 255), 2)
    lst_hash[0] -= 1
    collection.create_index([('hash', 'text')])
    cv.imwrite(f'{image_name}.png', img)
    return lst_hash


def identify_star(lst_hash):
    lst_lt, lst_gt = [0] * NUM_OF_CIRCLES, [0] * NUM_OF_CIRCLES
    lt_hash, gt_hash = '', ''
    for k in range(NUM_OF_CIRCLES):
        lt = lst_hash[k] - 2
        gt = lst_hash[k] + 1
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

    star_lst_hash_list = []
    for star in collection.find({'hash': {'$gt': lt_hash, '$lt': gt_hash}}):
        star_lst_hash = [0] * NUM_OF_CIRCLES

        for i in range(NUM_OF_CIRCLES):
            star_lst_hash[i] = int(star.get('hash')[(i * 2):2 + 2 * i])

        c = True
        for i in range(NUM_OF_CIRCLES):
            if lst_lt[i] <= star_lst_hash[i] <= lst_gt[i]:
                continue
            else:
                c = False
                break
        if c:
            star_lst_hash_list.append({star.get('_id'): star_lst_hash})

    min_distance = math.inf
    most_similar_id = None
    for item in star_lst_hash_list:
        for key, value in item.items():
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(lst_hash, value)))
            if distance < min_distance:
                min_distance = distance
                most_similar_id = key
    result_star = collection.find_one({'_id': most_similar_id})
    result_ra = result_star.get('location').get('coordinates')[0]
    result_dec = result_star.get('location').get('coordinates')[1]
    return [result_ra, result_dec]


def find_observer_coordinates(star1_rd, dist1_to_center_photo, star2_rd, dist2_to_center_photo,
                              star3_rd, dist3_to_center_photo, DEGREES_PER_PIXEL):
    earth_speed_per_second = 360 / (24 * 60 * 60)
    noon_today = datetime.datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    # seconds_since_noon = (datetime.datetime.now() - noon_today).total_seconds()    # num of sec from 12:00 to now
    seconds_since_noon = 0

    ro = earth_speed_per_second * seconds_since_noon     # the angle the Earth has turned from 12:00
    R = 6378.137

    latitude = star1_rd[0] - ro            # point on the Earth where star's light
    longitude = star1_rd[1]                # falls at a right angle
    observer_angle1 = dist1_to_center_photo * DEGREES_PER_PIXEL     # the angle at which the observer sees the star
    r01 = spherical_to_cartesian(longitude, latitude, R)

    latitude = star2_rd[0] - ro
    longitude = star2_rd[1]
    observer_angle2 = dist2_to_center_photo * DEGREES_PER_PIXEL
    r02 = spherical_to_cartesian(longitude, latitude, R)

    latitude = star3_rd[0] - ro
    longitude = star3_rd[1]
    observer_angle3 = dist3_to_center_photo * DEGREES_PER_PIXEL
    r03 = spherical_to_cartesian(longitude, latitude, R)

    A = np.array([r01, r02, r03])
    B = np.array([R * math.cos(math.radians(observer_angle1)),
                  R * math.cos(math.radians(observer_angle2)),
                  R * math.cos(math.radians(observer_angle3))])
    X = np.linalg.solve(A, B)

    length = math.sqrt(sum([component ** 2 for component in X]))
    res_x = X[0] / length
    res_y = X[1] / length
    res_z = X[2] / length

    observer_cartesian_coordinates = [res_x, res_y, res_z]
    observer_spherical_coordinates = cartesian_to_spherical(observer_cartesian_coordinates)
    return observer_spherical_coordinates


def run_program():
    import_data_from_csv_to_db()
    calculate_hash_for_db()
    generate_night_sky_photo('52146')
    image = cv.imread('night_sky_photo.png')
    DEGREES_PER_PIXEL = FIELD_OF_VIEW / image.shape[1]

    stars_coordinates_on_photo = find_stars(image)
    observer_coordinates = []
    triples_of_stars = select_stars_from_photo(stars_coordinates_on_photo, image)
    for i, triple_stars in enumerate(triples_of_stars):
        image2 = cv.imread('night_sky_photo.png')
        star1_hash = calculate_hash_for_star_photo(list(triple_stars[0].keys())[0], stars_coordinates_on_photo,
                                                   image2, DEGREES_PER_PIXEL, f'star_hash{i+1}')
        star2_hash = calculate_hash_for_star_photo(list(triple_stars[1].keys())[0], stars_coordinates_on_photo,
                                                   image2, DEGREES_PER_PIXEL, f'star_hash{i+1}')
        star3_hash = calculate_hash_for_star_photo(list(triple_stars[2].keys())[0], stars_coordinates_on_photo,
                                                   image2, DEGREES_PER_PIXEL, f'star_hash{i+1}')

        star1_rd = identify_star(star1_hash)
        star2_rd = identify_star(star2_hash)
        star3_rd = identify_star(star3_hash)

        observer_coordinates.append(find_observer_coordinates(star1_rd, list(triple_stars[0].values())[0],
                                                         star2_rd, list(triple_stars[1].values())[0], star3_rd,
                                                         list(triple_stars[2].values())[0], DEGREES_PER_PIXEL))
        print(f'\tЗнайдені координати {i+1}:', observer_coordinates[-1])
    total_latitude, total_longitude = 0, 0
    for pair in observer_coordinates:
        total_latitude += pair[0]
        total_longitude += pair[1]

    avg_latitude = total_latitude / len(observer_coordinates)
    avg_longitude = total_longitude / len(observer_coordinates)
    print('\nЗнайдені координати спостерігача: [', avg_latitude, ', ', avg_longitude, ']\n', sep='')


run_program()
