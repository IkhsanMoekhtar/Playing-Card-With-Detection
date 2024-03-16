import cv2
import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
import keyboard

def order_points(pts):
    # Mengurutkan titik berdasarkan koordinat x
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # Mendapatkan titik dengan koordinat x terkecil dan terbesar
    left = x_sorted[:2, :]
    right = x_sorted[2:, :]

    # Mengurutkan titik yang tersisa berdasarkan koordinat y
    left = left[np.argsort(left[:, 1]), :]
    (tl, bl) = left

    right = right[np.argsort(right[:, 1]), :]
    (tr, br) = right

    return np.array([tl, tr, br, bl], dtype="float32")

def warp_image(image, coords, precomputed_values=None):
    # Mengurutkan koordinat
    coords = order_points(coords)
    
    if precomputed_values is None:
        # Membuat matriks transformasi perspektif dan menerapkannya
        (tl, tr, br, bl) = coords
        widthA, widthB = np.linalg.norm(br - bl), np.linalg.norm(tr - tl)
        heightA, heightB = np.linalg.norm(tr - br), np.linalg.norm(tl - bl)
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    else:
        maxWidth, maxHeight, dst = precomputed_values

    M = cv2.getPerspectiveTransform(coords, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def load_template(template):
    tmpl = cv2.imread(os.path.join('C:/Users/ikhsa/OneDrive/Desktop/PCVTest/All_Remi_Card', template), cv2.IMREAD_GRAYSCALE)
    tmpl = cv2.GaussianBlur(tmpl, (5,5), 0)
    tmpl = cv2.adaptiveThreshold(tmpl, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 14)
    tmpl = cv2.erode(tmpl, None, iterations=2)
    tmpl = cv2.dilate(tmpl, None, iterations=1)
    return tmpl

def match_template(image_path, templates_folder, difference_threshold):
    kernel = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype=np.uint8)
    
    # Baca gambar input
    img = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 14)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=1)
    
    min_difference = float('inf')
    best_match = None
    
    # Dapatkan daftar semua file template dalam folder
    templates = os.listdir(templates_folder)
    
    # Muat semua template ke memori menggunakan multithreading
    with ThreadPoolExecutor(max_workers=8) as executor:
        loaded_templates = list(executor.map(load_template, templates))
    
    # Iterasi melalui setiap template yang dimuat
    for i, tmpl in enumerate(loaded_templates):
        
        # Jika ukuran template lebih besar dari gambar input, resize template
        if tmpl.shape[0] > img.shape[0] or tmpl.shape[1] > img.shape[1]:
            tmpl = cv2.resize(tmpl, (img.shape[1], img.shape[0]))
        
        # Cocokkan template dengan gambar input
        res = cv2.matchTemplate(img, tmpl, cv2.TM_SQDIFF)
        
        # Dapatkan nilai kemiripan maksimum
        min_val, _, _, _ = cv2.minMaxLoc(res)
        
        if min_val < min_difference:
            min_difference = min_val
            best_match = templates[i]
            
    return best_match, min_difference, img, tmpl

def pre_hsv(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    sensitivity = 100
    lower_white = np.array([0,0,255-sensitivity])
    upper_white = np.array([255,sensitivity,255])

    #img = cv2.GaussianBlur(img, (7,7), 0)
    mask_white = cv2.inRange(img, lower_white, upper_white)
    imgThreshold_white = cv2.erode(mask_white, None, iterations=2)
    imgThreshold_white2 = cv2.dilate(imgThreshold_white,None , iterations=1)
    imgThreshold_white3 = mask_white - (cv2.erode(imgThreshold_white2,None , iterations=1))

    return imgThreshold_white3

def count(coords , c_start , c_end ,detection_result):
    global frame,valid_contour_count, last_card_type,same_card_count,card_values,last_change_time
    
    if len(coords) == 4 and c_start[0] <= coords[0][0] <= c_end[0] and c_start[1] <= coords[0][1] <= c_end[1]:
        
        warped_image = warp_image(frame, np.array(coords, dtype="float32"))
        best_match,min_difference,tes,tmpl = match_template(warped_image, "C:/Users/ikhsa/OneDrive/Desktop/PCVTest/All_Remi_Card",0.8)
        
        card_type = os.path.splitext(best_match)[0]
        card_type_no_space = card_type.replace(" ", "")
        card_value = card_values.get(card_type_no_space, 0)
        
      
        cv2.putText(frame,f'{card_type}({card_value})'  , (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 8)
        cv2.putText(frame,f'{card_type}({card_value})' , (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 128), 4)
       
        valid_contour_count += 1
     
        current_time = time.time()
        #selecting and checking
        # Memilih dan memeriksa
        if last_card_type is None or last_card_type != card_type:
            last_card_type = card_type
            last_change_time = current_time  # Mengatur waktu perubahan terakhir
            same_card_count = 1
        else:
            same_card_count += 1
        
        
        if current_time - last_change_time >= 3:
            if (card_type, card_value) not in detection_result:
                detection_result.append((card_type, card_value))
        # Batasi ukuran detection_result untuk menghindari penggunaan memori yang berlebihan
                if len(detection_result) > 1000:
                    detection_result.pop(0)
            cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)
        else:
            card_detection_counts[card_type] = 0
            # Batasi ukuran card_detection_counts untuk menghindari penggunaan memori yang berlebihan
            if len(card_detection_counts) > 1000:
                oldest_card_type = min(card_detection_counts.keys(), key=(lambda k: card_detection_counts[k]))
                del card_detection_counts[oldest_card_type]
            
def result(detection_results, x, y, orientation='horizontal', line_height=50):
    global frame
    results_list = detection_results
    
    for i, result in enumerate(results_list):
        if result is not None:
            if orientation == 'horizontal':
                x_text = x + i*line_height
                y_text = y
            elif orientation == 'vertical':
                x_text = x
                y_text = y + i*line_height
            else:
                print("Invalid orientation. Please choose either 'horizontal' or 'vertical'.")
                return

            result_str = f'{result[0]}'
            if result_str:
                #cv2.putText(frame, result_str, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                card_image = cv2.imread(os.path.join('C:/Users/ikhsa/OneDrive/Desktop/PCVTest/All_Remi_Card', result_str + '.jpg'))
                # Mengubah ukuran gambar kartu agar sesuai dengan frame
                card_image = cv2.resize(card_image, (120, 170))
            
                # Menampilkan gambar kartu pada frame
                frame[y_text:y_text+170, x_text:x_text+120] = card_image
            else:
                print("result_str is empty.")
            

def find_target_suit(cards, card_values):
    
    if not cards:
        return None, None, None, None
    
    # Inisialisasi dictionary untuk menghitung jumlah kartu dari setiap jenis
    suit_counts = {'diamonds': 0, 'clubs': 0, 'hearts': 0, 'spades': 0}
    # Inisialisasi dictionary untuk menghitung nilai total dari setiap jenis kartu
    suit_values = {'diamonds': 0, 'clubs': 0, 'hearts': 0, 'spades': 0}
    
    # Iterasi melalui setiap kartu dalam list kartu
    for card in cards:
        # Jika kartu adalah tuple, proses sebagai tuple
        if isinstance(card, tuple):
            # Mengambil jenis dan nilai kartu dari card_values
            suit = card_values[card[0].strip()]['suit']
            value = card_values[card[0].strip()]['value']
            # Jika jenis kartu ada dalam suit_counts, tambahkan jumlah dan nilai kartu
            if suit in suit_counts:  # Memastikan bahwa jenis kartu valid
                suit_counts[suit] += 1
                suit_values[suit] += value

    # Mencari jenis kartu dengan nilai total terbesar
    target_suit = max(suit_values, key=suit_values.get)

    # Jika tidak ada jenis target (yaitu, tidak ada kartu), kembalikan None untuk semua output
    if target_suit is None:
        return None, None, None, None

    # Mengambil semua kartu dengan jenis target
    target_cards = [card for card in cards if isinstance(card, tuple) and card_values[card[0].strip()]['suit'] == target_suit]

    # Mengambil nilai dari setiap kartu target
    target_values = sum(card_values[card[0].strip()]['value'] for card in target_cards)
    
    # Menghitung total nilai kartu yang bukan target
    non_target_values = sum(card_values[card[0].strip()]['value'] for card in cards if isinstance(card, tuple) and card_values[card[0].strip()]['suit'] != target_suit)

    # Mengurangi total nilai kartu target dengan total nilai kartu yang bukan target
    target_values -= non_target_values

    # Kembalikan jenis target, kartu target, nilai target, dan nilai jenis
    return target_suit, target_cards, target_values, suit_values



# Fungsi untuk menentukan apakah kartu berhubungan dengan target komputer
def is_related_to_target(card, target):
    global player_detection_results,com_cards
    # Misalkan target adalah 'Hearts', maka kartu yang berhubungan adalah kartu dengan suit 'Hearts'
    return card == target





card_values = {
    '2d': {'value': 2, 'suit': 'diamonds'},
    '2c': {'value': 2, 'suit': 'clubs'},
    '2h': {'value': 2, 'suit': 'hearts'},
    '2s': {'value': 2, 'suit': 'spades'},
    '3d': {'value': 3, 'suit': 'diamonds'},
    '3c': {'value': 3, 'suit': 'clubs'},
    '3h': {'value': 3, 'suit': 'hearts'},
    '3s': {'value': 3, 'suit': 'spades'},
    '4d': {'value': 4, 'suit': 'diamonds'},
    '4c': {'value': 4, 'suit': 'clubs'},
    '4h': {'value': 4, 'suit': 'hearts'},
    '4s': {'value': 4, 'suit': 'spades'},
    '5d': {'value': 5, 'suit': 'diamonds'},
    '5c': {'value': 5, 'suit': 'clubs'},
    '5h': {'value': 5, 'suit': 'hearts'},
    '5s': {'value': 5, 'suit': 'spades'},
    '6d': {'value': 6, 'suit': 'diamonds'},
    '6c': {'value': 6, 'suit': 'clubs'},
    '6h': {'value': 6, 'suit': 'hearts'},
    '6s': {'value': 6, 'suit': 'spades'},
    '7d': {'value': 7, 'suit': 'diamonds'},
    '7c': {'value': 7, 'suit': 'clubs'},
    '7h': {'value': 7, 'suit': 'hearts'},
    '7s': {'value': 7, 'suit': 'spades'},
    '8d': {'value': 8, 'suit': 'diamonds'},
    '8c': {'value': 8, 'suit': 'clubs'},
    '8h': {'value': 8, 'suit': 'hearts'},
    '8s': {'value': 8, 'suit': 'spades'},
    '9d': {'value': 9, 'suit': 'diamonds'},
    '9c': {'value': 9, 'suit': 'clubs'},
    '9h': {'value': 9, 'suit': 'hearts'},
    '9s': {'value': 9, 'suit': 'spades'},
    '10d': {'value': 10, 'suit': 'diamonds'},
    '10c': {'value': 10, 'suit': 'clubs'},
    '10h': {'value': 10, 'suit': 'hearts'},
    '10s': {'value': 10, 'suit': 'spades'},
    'jd': {'value': 10, 'suit': 'diamonds'},
    'jc': {'value': 10, 'suit': 'clubs'},
    'jh': {'value': 10, 'suit': 'hearts'},
    'js': {'value': 10, 'suit': 'spades'},
    'qd': {'value': 10, 'suit': 'diamonds'},
    'qc': {'value': 10, 'suit': 'clubs'},
    'qh': {'value': 10, 'suit': 'hearts'},
    'qs': {'value': 10, 'suit': 'spades'},
    'kd': {'value': 10, 'suit': 'diamonds'},
    'kc': {'value': 10, 'suit': 'clubs'},
    'kh': {'value': 10, 'suit': 'hearts'},
    'ks': {'value': 10, 'suit': 'spades'},
    'ad': {'value': 11, 'suit': 'diamonds'},
    'ac': {'value': 11, 'suit': 'clubs'},
    'ah': {'value': 11, 'suit': 'hearts'},
    'as': {'value': 11, 'suit': 'spades'}
}


cap =cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX 
contour_count = 0
prev_contours = []
com_detection_results = []
player_detection_results = []
card_detection_counts = {}
last_card_type = None
same_card_count = 0
deck_detection_results =[]
comThrow_detection_results = []
final_detection_results = []
has_discarded = False


last_change_time = None


# Mengatur lebar frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

# Mengatur tinggi frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Inisialisasi variabel
start_detection = False

text_game = "GILIRAN PEMAIN"

toggle_detection = False  # variabel toggle

#input
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    
    HSV_pre = pre_hsv(frame)
    
    contours, _= cv2.findContours(HSV_pre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    valid_contour_count = 0
    current_contours = [] 
    
    

    # player
    com_start = (950, 20)  # Titik awal (x, y)
    com_end = (1250, 330)  # Titik akhir (x, y)
    
    player_start = (30,20)
    player_end   = (340,330)
    
    comThrow_start =  (30, 370)
    comThrow_end = (330, 700)
    
    
    throw_start = (480,370) 
    throw_end = (800,680)

    final_start = (480,20)
    final_end   = (800,350)

    
    
    for cnt in contours : 
        area = cv2.contourArea(cnt)

        if start_detection:
            if area > 6000 :
                epsilon = (0.06) * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True) 
                
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  
                
                n = approx.ravel()  
                i = 0
                coords = []
                     
                
                for j in n : 
                    if(i % 2 == 0): 
                        x = n[i] 
                        y = n[i + 1] 
                        
                        # String containing the co-ordinates. 
                        #string = str(x) + "," + str(y)  
                        coords.append((x, y))
                    i = i + 1
               
                # Calculate the center of the contour
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])   
            
                #mencari hasil
                com_count= count(coords, com_start, com_end, com_detection_results)
                player_count = count(coords, player_start, player_end, player_detection_results)
                deck_count = count(coords, throw_start, throw_end, deck_detection_results)
                comThrow_count = count(coords, comThrow_start, comThrow_end, comThrow_detection_results)
               
                
    result(com_detection_results,960,340)
    result(player_detection_results,360,30)
    result(deck_detection_results, 500, 300)
    result(comThrow_detection_results, 70, 380,'vertical',0)
    #print(com_detection_results)
    
    com_suit, com_cards, com_values, com_suit_values = find_target_suit(com_detection_results, card_values)
    # Menentukan teks dan posisi
    texts = [
    f"Target suit: {com_suit}",
    f"Target cards: {com_cards}",
    f"Target values: {com_values}",
    f"Suit values: {com_suit_values}"
    ]


    vWhite = np.zeros((100 , frame.shape[1], frame.shape[2]), np.uint8)
    vWhite.fill(255)
    frame = cv2.vconcat([frame, vWhite])

#score========================================================================
    com_value = f"score: {com_values}"
    # Menambahkan setiap teks ke gambar
    #for i, text in enumerate(com_value):
    position = (1100,770 )  # posisi teks (x, y)
    cv2.putText(frame, com_value, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
# com algorithm===============================================================
    
        
    # Menghitung lebar teks
    textsize = cv2.getTextSize(text_game, font, 1, 2)[0]

    


    # Menentukan posisi teks
    textX = (frame.shape[1] - textsize[0]) // 2
    textY = (frame.shape[0] + textsize[1]) // 2

    if len(com_detection_results) >= 4:
        cv2.putText(frame, text_game,(textX,775), font, 1, (0, 0, 0), 5)
        cv2.putText(frame, text_game,(textX,775), font, 1, (255, 255, 255), 2)
        
    card = None
    has_drawn = False  # Variabel baru

    # Jika ada hasil deteksi pemain
    if len(player_detection_results) > 0 :
        is_related = is_related_to_target(player_detection_results[0][1]['suit'], com_cards[0][1]['suit'])
        if is_related:  
            if player_detection_results:  # Cek jika daftar tidak kosong
                card = player_detection_results[0]  # Dapatkan kartu pertama dari player_detection_result
                min_card = min(com_detection_results, key=lambda c: c[1]['value'])
                print('tes')
                if is_related and card[1]['value'] <  min_card[1]['value'] and len(set(card[1]['suit'] for card in com_detection_results)) == 1:
                    text_game = "Player don't took the card from player throw"
                    is_related = False
                    if len(deck_detection_results) > 0:  
                        # Mengambil kartu dari deck
                        if deck_detection_results:
                            card = deck_detection_results.pop(0)
                            # Menampilkan pesan bahwa komputer mengambil kartu dari deck
                            print(f"Computer drew {card} from the deck.")
                            text_game =f"Computer drew {card[0]} from the deck."
                            textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                            #cv2.putText(frame, f"Computer drew {card} from the deck.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
                            #cv2.putText(frame, f"Computer drew {card} from the deck.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            # Menandai bahwa komputer telah mengambil kartu
                            has_drawn = True
                            deck_detection_results = []
                            player_detection_results = []
                            com_detection_results.append(card)
                  
                            #if not has_drawn:  # Cek jika belum melakukan append
                                #com_detection_results.append(card)
                                #has_drawn = True  # Menandai bahwa append telah dilakukan
                            start_detection = False
                            
                            # Jika semua kartu dalam com_detection_results memiliki jenis yang sama
                            if len(set(card[1]['suit'] for card in com_detection_results)) == 1:
                                
                                min_card = min(com_detection_results, key=lambda c: c[1]['value'])
                                # Hapus kartu dengan nilai terkecil dari com_detection_results
                                com_detection_results.remove(min_card)
                                comThrow_detection_results.append(min_card)

                else:
                    print(f"Computer took {card} from the player's discard pile.")
                    text_game = f"Computer took {card[0]} from the player's discard pile."
                    textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                    has_drawn = True  
                    player_detection_results = []
                    com_detection_results.append(card)
                    deck_detection_results = []
                    if not has_drawn:  # Cek jika belum melakukan append
                        com_detection_results.append(card)
                        has_drawn = True  # Menandai bahwa append telah dilakukan
                    start_detection = False 
        
                    non_target_cards = [c for c in com_detection_results if not is_related_to_target(c[1]['suit'], com_cards[0][1]['suit'])]
                      # Jika ada kartu dengan jenis yang tidak sesuai dengan target
                    if non_target_cards:
                        # Menyimpan kartu dari tumpukan buangan pemain dan membuang kartu dengan nilai tertinggi dan jenis yang tidak sesuai dengan target
                        max_card = max(non_target_cards, key=lambda c: c[1]['value'])  
                        com_detection_results.remove(max_card)
                        comThrow_detection_results.append(max_card)
                        print(f"Computer discarded {max_card} from its hand.")
                        text_game = f"Computer discarded {max_card[0]} from its hand."
                        textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                    else :
                        
                        com_detection_results.remove(min_card)
                        comThrow_detection_results.append(min_card)
                        text_game = f"Computer discarded {min_card[0]} from its hand."
                        textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                        
                    
                # Jika kartu dari tumpukan buangan pemain tidak sesuai dengan target
            if not is_related:
                # Mencari kartu dengan jenis yang tidak sesuai dengan target di tangan komputer
                non_target_cards = [c for c in com_detection_results if not is_related_to_target(c[1]['suit'], com_cards[0][1]['suit'])]
                # Jika ada kartu dengan jenis yang tidak sesuai dengan target
                if non_target_cards:
                    # Mencari kartu dengan nilai terkecil di antara kartu dengan jenis yang tidak sesuai dengan target
                    min_card = min(non_target_cards, key=lambda c: c[1]['value'])  
                    # Jika kartu dari tumpukan buangan pemain memiliki nilai lebih kecil atau sama dengan kartu dengan nilai terkecil di antara kartu dengan jenis yang tidak sesuai dengan target
                    if card[1]['value'] <= min_card[1]['value']:  
                        # Membuang kartu dari tumpukan buangan pemain
                        com_detection_results.remove(min_card)
                        print(f"Computer discarded {card} from its hand.")
                        text_game = f"Computer discarded {card[0]} from its hand."
                        textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                        comThrow_detection_results.append(min_card)
                    else:
                        # Menyimpan kartu dari tumpukan buangan pemain dan membuang kartu dengan nilai tertinggi dan jenis yang tidak sesuai dengan target
                        max_card = max(non_target_cards, key=lambda c: c[1]['value'])  
                        com_detection_results.remove(max_card)
                        print(f"Computer discarded {max_card} from its hand.")
                        text_game = f"Computer discarded {max_card[0]} from its hand."
                        textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                        com_detection_results.append(card)
        else:
            # Jika ada kartu di deck dan komputer belum mengambil kartu
            if len(deck_detection_results) > 0:  
                # Mengambil kartu dari deck
                if deck_detection_results:
                    card = deck_detection_results.pop(0)
                    # Menampilkan pesan bahwa komputer mengambil kartu dari deck
                    print(f"Computer drew {card} from the deck.")
                    text_game =f"Computer drew {card[0]} from the deck."
                    textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                    #cv2.putText(frame, f"Computer drew {card} from the deck.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
                    #cv2.putText(frame, f"Computer drew {card} from the deck.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # Menandai bahwa komputer telah mengambil kartu
                    has_drawn = True
                    deck_detection_results = []
                    player_detection_results = []
                    com_detection_results.append(card)
          
                    #if not has_drawn:  # Cek jika belum melakukan append
                        #com_detection_results.append(card)
                        #has_drawn = True  # Menandai bahwa append telah dilakukan
                    start_detection = False
                    
                    # Jika semua kartu dalam com_detection_results memiliki jenis yang sama
                    if len(set(card[1]['suit'] for card in com_detection_results)) == 1:
                        
                        min_card = min(com_detection_results, key=lambda c: c[1]['value'])
                        # Hapus kartu dengan nilai terkecil dari com_detection_results
                        com_detection_results.remove(min_card)
                        comThrow_detection_results.append(min_card)
                    
                    # Jika kartu dari deck tidak sesuai dengan target
                    non_target_cards = [c for c in com_detection_results if not is_related_to_target(c[1]['suit'],com_cards[0][1]['suit'] )]
                    # Jika ada kartu dengan jenis yang tidak sesuai dengan target
                    if non_target_cards:
                        max_card = max(non_target_cards, key=lambda c: c[1]['value'])  
                        com_detection_results.remove(max_card)
                        comThrow_detection_results.append(max_card)
                        print(f"Computer discarded {max_card} from its hand.")
                        text_game = f"Computer discarded {max_card[0]} from its hand."
                        textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                        #cv2.putText(frame, f"Computer discarded {max_card} from its hand.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
                        #cv2.putText(frame, f"Computer discarded {max_card} from its hand.",(520,775), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                    if not is_related_to_target(card[1]['suit'], com_cards[0][1]['suit']):  
                        # Mencari kartu dengan jenis yang tidak sesuai dengan target di tangan komputer
                        non_target_cards = [c for c in com_detection_results if not is_related_to_target(c[1]['suit'],com_cards[0][1]['suit'] )]
                        # Jika ada kartu dengan jenis yang tidak sesuai dengan target
                        if non_target_cards:
                            # Mencari kartu dengan nilai terkecil di antara kartu dengan jenis yang tidak sesuai dengan target
                            min_card = min(non_target_cards, key=lambda c: c[1]['value'])
                            
                            # Jika kartu dari deck memiliki nilai lebih kecil atau sama dengan kartu dengan nilai terkecil di antara kartu dengan jenis yang tidak sesuai dengan target
                            #if card[1]['value'] <= min_card[1]['value']:  
                                #com_detection_results.append(card)
                else:
                    # Menampilkan pesan bahwa deck kosong
                    print("Deck is empty.")
                    text_game = "Deck is empty."
                    textsize = cv2.getTextSize(text_game, font, 1, 2)[0]
                    
#end of com algorithm=========================================================               


    if com_values == 41:
        text_game = "player win"

    if keyboard.is_pressed('t'):  # Jika tombol 't' ditekan
        if comThrow_detection_results:  # Jika list tidak kosong
            card = comThrow_detection_results.pop()  # Menghapus item terakhir dari list
            print(f"Removed {card} from comThrow_detection_results")
        else:
            print("comThrow_detection_results is empty.")
        
        
    if keyboard.is_pressed('r'):
        deck_detection_results = []
        player_detection_results = []
        comThrow_detection_results = []

    if keyboard.is_pressed('a'):
        com_detection_results = [('jd ', {'value': 10, 'suit': 'diamonds'}),
                                 ('kh ', {'value': 10, 'suit': 'diamonds'}),
                                 ('10d ',{'value': 10, 'suit': 'diamonds'}),
                                 ('as ', {'value': 11, 'suit': 'spades'})]
   
    

    
    
    
    if keyboard.is_pressed('c'):
        toggle_detection = not toggle_detection  # ubah status deteksi
    
    if toggle_detection:
        for cnt in contours : 
            area = cv2.contourArea(cnt)

            if area > 6000 :
                epsilon = (0.06) * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True) 
                
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)  
                
                n = approx.ravel()  
                i = 0
                coords = []
                
                
                for j in n : 
                    if(i % 2 == 0): 
                        x = n[i] 
                        y = n[i + 1] 
                        
                        # String containing the co-ordinates. 
                        #string = str(x) + "," + str(y)  
                        coords.append((x, y))
                    i = i + 1
               
                # Calculate the center of the contour
                M = cv2.moments(cnt)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])   
                
                final_count = count(coords, final_start, final_end, final_detection_results)
                
        frame = cv2.rectangle(frame, final_start, final_end, (0, 0, 255), 2)


    final_suit, final_cards, final_values, final_suit_values = find_target_suit(final_detection_results, card_values) 
    w_pressed = False
    exit_loop = False
    
    if com_values == 41 or (not w_pressed and keyboard.is_pressed('w')):
        w_pressed = True  # Ubah variabel ini menjadi True setelah 'w' ditekan
        text_end = ["Game Over!!", "Computer Win"]
        font = cv2.FONT_HERSHEY_SIMPLEX
    
        for i in range(1, 11):
            alpha = i / 10.0  # alpha berkisar dari 0.1 hingga 1.0
            black_frame = np.zeros_like(frame)
            
            for j, line in enumerate(text_end):
                textsize = cv2.getTextSize(line, font, 1, 2)[0]
                textX = (black_frame.shape[1] - textsize[0]) // 2  # posisi tengah teks di sumbu x
                textY = ((black_frame.shape[0] + textsize[1]) // 2) + (j * 40)  # posisi tengah teks di sumbu y, ditambah dengan jarak antar baris
                cv2.putText(black_frame, line, (textX, textY), font, 1, (255, 255, 255), 2)
    
            # Membuat frame yang merupakan campuran antara frame asli dan frame hitam
            blended = cv2.addWeighted(frame, 1 - alpha, black_frame, alpha, 0)
            cv2.imshow('Frame', blended)
            cv2.waitKey(100)  # Tunda untuk 100 ms untuk mengendalikan kecepatan animasi
    
        while True:
            cv2.imshow('Frame', black_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar dari loop
                exit_loop = True
                break
    
        if exit_loop:
            break


    elif final_values is not None and com_values is not None:
        if len(final_detection_results) >= 4:
            if final_values > com_values:
                w_pressed = True  # Ubah variabel ini menjadi True setelah 'w' ditekan
                text_end = ["Game Over!!", "player Win"]
                font = cv2.FONT_HERSHEY_SIMPLEX
            
                for i in range(1, 11):
                    alpha = i / 10.0  # alpha berkisar dari 0.1 hingga 1.0
                    black_frame = np.zeros_like(frame)
                    
                    for j, line in enumerate(text_end):
                        textsize = cv2.getTextSize(line, font, 1, 2)[0]
                        textX = (black_frame.shape[1] - textsize[0]) // 2  # posisi tengah teks di sumbu x
                        textY = ((black_frame.shape[0] + textsize[1]) // 2) + (j * 40)  # posisi tengah teks di sumbu y, ditambah dengan jarak antar baris
                        cv2.putText(black_frame, line, (textX, textY), font, 1, (255, 255, 255), 2)
            
                    # Membuat frame yang merupakan campuran antara frame asli dan frame hitam
                    blended = cv2.addWeighted(frame, 1 - alpha, black_frame, alpha, 0)
                    cv2.imshow('Frame', blended)
                    cv2.waitKey(100)  # Tunda untuk 100 ms untuk mengendalikan kecepatan animasi
            
                while True:
                    cv2.imshow('Frame', black_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar dari loop
                        exit_loop = True
                        break
            
                if exit_loop:
                    break
            elif final_values < com_values:
                w_pressed = True  # Ubah variabel ini menjadi True setelah 'w' ditekan
                text_end = ["Game Over!!", "Computer Win"]
                font = cv2.FONT_HERSHEY_SIMPLEX
            
                for i in range(1, 11):
                    alpha = i / 10.0  # alpha berkisar dari 0.1 hingga 1.0
                    black_frame = np.zeros_like(frame)
                    
                    for j, line in enumerate(text_end):
                        textsize = cv2.getTextSize(line, font, 1, 2)[0]
                        textX = (black_frame.shape[1] - textsize[0]) // 2  # posisi tengah teks di sumbu x
                        textY = ((black_frame.shape[0] + textsize[1]) // 2) + (j * 40)  # posisi tengah teks di sumbu y, ditambah dengan jarak antar baris
                        cv2.putText(black_frame, line, (textX, textY), font, 1, (255, 255, 255), 2)
            
                    # Membuat frame yang merupakan campuran antara frame asli dan frame hitam
                    blended = cv2.addWeighted(frame, 1 - alpha, black_frame, alpha, 0)
                    cv2.imshow('Frame', blended)
                    cv2.waitKey(100)  # Tunda untuk 100 ms untuk mengendalikan kecepatan animasi
            
                while True:
                    cv2.imshow('Frame', black_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar dari loop
                        exit_loop = True
                        break
            
                if exit_loop:
                    break
            else:
                print("final_values atau com_values adalah None")


    if len(com_detection_results) >= 4:
        final_text = "press c if you already have 41 value of card"
        # Menambahkan setiap teks ke gambar
        #for i, text in enumerate(com_value):
        position_final = (380,30 )  # posisi teks (x, y)
        cv2.putText(frame, final_text, position_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the number of valid contours on the frame
    text = "Card Detected: {}".format(valid_contour_count)
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    textX = (frame.shape[1] - textsize[0]) // 2
    textY = frame.shape[0] - 120
    cv2.putText(frame, text, (textX, textY), font, 1, (0, 255, 0), 2)
    

    
    frame = cv2.rectangle(frame, com_start, com_end, (0, 255, 0), 2)
    frame = cv2.rectangle(frame, player_start, player_end, (255, 255, 0), 2)
    frame = cv2.rectangle(frame, throw_start, throw_end, (0, 255, 255), 2)
    #frame = cv2.rectangle(frame, comThrow_start, comThrow_end, (255, 255,0 ), 2)
    
    
    #frame = cv2.rectangle(frame, (640,720), (640,820), (0, 0, 0), 10)
    cv2.line(frame, (210, 725), (210, 820), (0, 0, 0), 10)
    cv2.line(frame, (1070, 725), (1070, 820), (0, 0, 0), 10)
    cv2.line(frame, (0, 820), (1280, 820), (0, 0, 0), 20)       
    cv2.imshow('Frame', frame)
    #cv2.imshow('image', imgThreshold_white3)  
        
        
                    
                    
    key = cv2.waitKey(1) & 0xFF

    # Menunggu tombol spasi ditekan
    if key == ord(' '):
        start_detection = not start_detection
    elif key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
