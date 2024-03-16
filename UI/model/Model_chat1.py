import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re


def replace_words(text):
    # Danh sách từ cần thay đổi và từ mới
    replacements = {'\\blă\b': ' lắm ', '\\bno\\b': ' nó ', '\\bbi\b': ' bị ','\\btrất\\b': ' tốt nhất ',
                    '\\bt\b': ' tao ', '( y )': ' positive ', '\\bbjo\\b': ' bây giờ ', "\\blz\\b": "lồn", "\\bDKMM\\b": "đm",
                    '\\bsút\\b': ' súc ', '\\bne\\b': ' nè ', '\\bchừ\\b': ' giờ ', "5 cu ( củ )":"5 triệu",
                    '\\bhs\\b': ' học sinh ', '\\bsgk\\b': ' sách giáo khoa ', '\\bcv\\b': ' công việc ', "\\bper\\b": "",
                    '\\blink\\b': ' đường dẫn ', '\\bkq\\b': ' kết quả ', '\\bdzầy\\b': ' này ', "\\bcc\b": "con cặc",
                    '\\b5tr\\b': ' 5 triệu ', '\\bc thy\\b': ' chị thy ', '\\bad\\b': ' quản trị viên ',
                    '\\bz\\b': ' vậy ', '\\bcta\\b': ' chúng ta ', '\\bks\\b': ' khách sạn ', '\\bafk\\b': ' à ',
                    '\\bk\\b': ' không ', '\\bm\\b': ' mày ', '\\bt\\b': ' tao ', '\\bfa\\b': ' cô đơn ',
                    '\\bnyc\\b': ' người yêu cũ ', '\\bnge\\b': ' nghe ', '\\bzi\\b': ' vậy ',"\\bDM\\b": "đm", "\\bĐM\\b": "đm",
                    '\\bcủng\\b': ' cũng ', '\\bsứng\\b': ' xứng ', '\\bof\\b': ' của ', '\\bchay\\b': ' chạy ',
                    '\\bkip\\b': ' kịp ', '\\bg\\b': ' gì ', '\\bchowi\\b': ' chơi ', '\\btúm\\b': ' tụm ',
                    '\\bstt\\b': ' trạng thái ', '\\blò xô\\b': ' lò xo ', '\\bfải\\b': ' phải ',
                    '\\bviecj\\b': ' việc ', '\\bma\\b': ' mà ', '\\bláp\\b': ' láo ', '\\bzo\\b': ' vô ',
                    '\bhah\b': ' ha ha','\bhaa\b':'ha ha', '\\bq.tâm\\b': ' quan tâm ','\\bpr\\b': ' quảng cáo ',
                    '\\bơiii\\b': ' ơi ', '\\bdume\\b': ' đm ', '\\bchớt\\b': ' chết ', "\\bc * c\\b": "cặc",
                    '\\bcrush\\b': ' người mình thích ', '\\bnta\\b': ' người ta ', "\\bdt\\b": "điện thoại",
                    '\\bthậc\\b': ' thật ', '\\blsao\\b': ' làm sao ', '\\bmặp\\b': ' mập ', "\\b:d\\b" : ":D",
                    '\\bcờ rút\\b': ' người mình thương ', '\\bdồi\\b': ' rồi ', '\\bnghiê\\b': ' nghiệp ',
                    '\\btưở\\b': ' tưởng ', '\\bbướ c\\b': ' bước ', "\\bmuốn bắn , muốn làm việc\\b": "không muốn bắn, không muốn làm việc",
                    '\\bhaizzz\\b': ' buồn ', '\\bluônnn\\b': ' luôn ', "\\bkì thích\\b" : "cứ thích",
                    '\\btui thi ́ ch va ̉ i  lắm  ́ mày mà ̀ ăn nhỉ ̀ u  nó  ́ người  lắm  ́ mày mọi người anh ̣ ,  bị  ̣ lơ ̉ miê ̣ người\\b': "tôi thích vải lắm mà ăn nhiều nóng người lắm mọi người à, bị lỡ miệng ấy",
                    '\\bkkk\\b': ' vui ', '\\bkhác hàg\\b': ' khách hàng ', '\\bđanh\\b': ' đánh ',
                    '\\bgym\\b': ' phòng thể dục ', '\\btime\\b': ' thời gian ', '\\bmóa\\b': ' má ',
                    '\\bmink\\b': ' mình ', '\\bnv\\b': ' như vậy ', '\\bchư\\b': ' chứ ', "\\bghê răng 😁\\b": "ghê quá",
                    '\\bvãi_chưởng\\b': ' vãi chưởng ', '\\bthuờng\\b': 'bình thường', '\\boi\\b': ' ơi ','\\blă\\b': ' lắm ', '\\bno\\b': ' nó ',
                    '\\bbi\\b': ' bị ', '\\btrất\\b': ' tốt nhất ','\\bhs\\b': 'học sinh', '\\bsgk\\b': 'sách giáo khoa', '\\bcv\\b': 'công việc',
                    '\\bdzầy\\b': 'này', '\\b5tr\\b': '5 triệu', '\\bc thy\\b': 'chị thy', '\\bad\\b': 'quản trị viên', '\\bz\\b': 'vậy',
                    '\\bcta\\b': 'chúng ta', '\\bks\\b': 'khách sạn', '\\bafk\\b': 'à', '\\bk\\b': 'không', '\\bm\\b': 'mầy', '\\bt\\b': 'tao',
                    '\\bfa\\b': 'cô đơn', '\\bnyc\\b': 'người yêu cũ', '\\bnge\\b': 'nghe', '\\bzi\\b': 'vậy', '\\bcủng\\b': 'cũng', '\\bsứng\\b': 'xứng',
                    '\\bof\\b': 'của', '\\bchay\\b': 'chạy', '\\bkip\\b': 'kịp', '\\bg\\b': 'gì', '\\bchowi\\b': 'chơi', '\\btúm\\b': 'tụm',
                    '\\bstt\\b': 'trạng thái', '\\blò xô\\b': 'lò xo', '\\bfải\\b': 'phải', '\\bviecj\\b': 'việc', '\\bma\\b': 'mà',
                    '\\bláp\\b': 'láo', '\\bzo\\b': 'vô', '\\bhihi\\b': 'hi hi', '\\bhahah\\b': 'ha ha', '\\bq.tâm\\b': 'quan tâm', '\\bhuhu\\b': 'hu hu',
                    '\\bơiii\\b': 'ơi', '\\bpr\\b': 'quảng cáo', '\\bcrush\\b': 'người mình thích', '\\bnta\\b': 'người ta',
                    '\\bchớt\\b': 'chết', '\\bthậc\\b': 'thật', '\\blsao\\b': 'làm sao', '\\bmặp\\b': 'mập', '\\bcờ rút\\b': 'người mình thương', '\\bdồi\\b': 'rồi',
                    '\\bnghiê\\b': 'nghiệp', '\\btưở\\b': 'tưởng', '\\bbướ c\\b': 'bước', '\\bhaha\\b': 'ha ha', '\\bhaizzz\\b': 'buồn',
                    '\\bhí hí\\b': 'hi hi', '\\bvkl\\b': 'vui', '\\bkkk\\b': 'ha ha', '\\bkhác hàg\\b': 'khách hàng', '\\bluônnn\\b': 'luôn',
                    '\\bgym\\b': 'phòng thể dục', '\\btime\\b': 'thời gian', '\\bmóa\\b': 'má', '\\bmink\\b': 'mình', '\\bnv\\b': 'như vậy', '\\bchư\\b': 'chứ',
                    '\\bđanh\\b': 'đánh', '\\bvãi_chưởng\\b': 'vãi chưởng', '\\bthuờng\\b': 'thường', '\\boi\\b': 'ơi', '\\btroi\\b': 'trời', '\\brả\\b': 'giả',
                    '\\bcờ rớt\\b': 'người mình thương', '\\bcờ hó\\b': 'chó', '\\bhahahaha\\b': 'ha ha', '\\be\\b': 'em',
                    '\\bthuoc\\b': 'thuộc', '\\bnua\\b': 'nữa', '\\bbat\\b': 'bắt', '\\bkiem tien\\b': 'kiếm tiền', '\\bchờiiiii\\b': 'trời',
                    '\\bpage\\b': 'trang', '\\bshipper\\b': 'người giao hàng', '\\bquáaaaa\\b': 'quá', '\\btôiii\\b': 'tôi', '\\bs\\b': 'sao', '\\bz\\b': 'vậy',
                    '\\bdel\\b': 'đéo', '\\br\\b': 'rồi', '\\bcute\\b': 'dễ thương', '\\bđéo bít\\b': 'không biết', '\\bah\\b': 'ạ',
                    '\\bđấyyyy\\b': 'đấy', '\\bth\\b': 'thằng', '\\bcám\\b': 'cảm', '\\bvn\\b': 'việt nam', '\\bng\\b': 'người', '\\byêuc\\b': 'yêu',
                    '\\bhuhuhu\\b': 'buồn', '\\bntnày\\b': 'như thế này', '\\bpv\\b': 'phục vụ', '\\bmún\\b': 'muốn', '\\btroll\\b': 'đùa',
                    '\\bcf\\b': 'cà phê', '\\bthôiiii\\b': 'thôi', '\\bthg\\b': 'thằng', '\\bdth\\b': 'dễ thương', '\\bhv\\b': 'như vậy', '\\bổng\\b': 'ông',
                    '\\bdell\\b': 'đéo', '\\bđâyy\\b': 'đây', '\\bkkk\\b': 'vui', '\\bzai\\b': 'trai', '\\bquad\\b': 'quá',
                    '\\bbabe\\b': 'baby', '\\b1b trai\\b': 'một bạn trai', '\\bđcm\\b': 'đm', '\\bgg\\b': 'google',
                    '\\bstk\\b': 'số tài khoản', '\\bcsong\\b': 'cuộc sống', '\\bko\\b': 'không', '\\bc trai\\b': 'con trai',
                    '\\bđíuu\\b': 'đéo', '\\bcsgt\\b': 'cảnh sát giao thông', '\\bhaha\\b': 'ha ha', '\\be\\b': 'em', "\\bcute\\b": "dễ thương",
                    '\\bthuoc\\b': 'thuộc', '\\bnua\\b': 'nữa', '\\bbat\\b': 'bắt', '\\bkiem tien\\b': 'kiếm tiền', '\\bchờiiiii\\b': 'trời',
                    '\\bpage\\b': 'trang', '\\bshipper\\b': 'người giao hàng', '\\bquáaaaa\\b': 'quá', '\\btôiii\\b': 'tôi', '\\bs\\b': 'sao',
                    '\\bz\\b': 'vậy', '\\br\\b': 'rồi', '\\bcute\\b': 'dễ thương', '\\bđéo bít\\b': 'đéo biết', '\\bdcm\\b': 'đm',
                    '\\bah\\b': 'ạ', '\\bđấyyyy\\b': 'đấy', '\\bth\\b': 'thằng', '\\bcám\\b': 'cảm', '\\bvn\\b': 'Việt Nam', '\\bng\\b': 'người', '\\byêuc\\b': 'yêu',
                    '\\bhuhuhu\\b': 'buồn', '\\bntnày\\b': 'như thế này', '\\bpv\\b': 'phục vụ', '\\bmún\\b': 'muốn', '\\btroll\\b': 'đùa',
                    '\\bcf\\b': 'cà phê', '\\bthôiiii\\b': 'thôi', '\\bthg\\b': 'thằng', '\\bdth\\b': 'dễ thương', '\\bhv\\b': 'như vậy', '\\bổng\\b': 'ông',
                    '\\bđâyy\\b': 'đây', '\\bkkk\\b': 'vui', '\\bzai\\b': 'trai', '\\bquad\\b': 'quá', "\\bcụ\\b": "mày", "\\bCụ\\b" : "mày",
                    '\\bdkm\\b': 'đm', '\\bbabe\\b': 'baby', '\\b1b trai\\b': 'một bạn trai','\\bđcm\\b': ' đm ', '\\bgg\\b': ' google ',
                    '\\bstk\\b': ' số tài khoản ', '\\bcsong\\b': ' cuộc sống ', '\\bko\\b': ' không ', '\\bdisme\\b': ' nagative ', '\\bc trai\\b': ' con trai ',
                    '\\bđíuu\\b': ' đéo ', '\\bxạolin\\b': ' xạo lồn ', '\\bcsgt\\b': ' cảnh sát giao thông ', '\\bhix hix\\b': ' buồn ',
                    '\\bđiiii\\b': ' đi ', '\\bhix\\b': ' buồn ', '\\bcam on\\b': ' cảm ơn ', '\\bmịe\\b': ' buồn ', '\\bthíck\\b': ' thích ',
                    '\\bdisss\\b': ' địt ', '\\bàk\\b': ' à ', '\\bvãiii\\b': ' vãi ', '\\bdì\\b': ' gì ', '\\bchộm\\b': ' trộm ', '\\bcéc\\b': ' cặc ',
                    '\\bhaaaaa\\b': ' ha ha ', '\\bvầy\\b': ' như này ', '\\b20 - 30\\b': ' hai mươi đến ba mươi ', '\\bcute\\b': ' dễ thương ',
                    '\\bcđv\\b': ' cỗ động viên ', '\\bmềnh\\b': ' mình ', '\\bnhể\\b': ' nhỉ ', '\\bdrama\\b': ' kịch ', '\\bgato\\b': ' ganh tị ',
                    '\\b1\\b': ' một ', '\\bphắng\\b': ' phắn ', '\\bhêtd\\b': ' hết ', '\\bquay lip\\b': ' quay clip ', '\\brướn\\b': ' rướm ', '\\bhjhj\\b': ' hi hi ',
                    '\\bcta\\b': ' chúng ta ', '\\bhnua\\b': ' hôm nữa ', '\\bfull\\b': ' đầy ', '\\bzai\\b': ' trai ', '\\b400k\\b': ' bốn trăm nghìn ',
                    '\\byêuc\\b': ' yêu ', '\\bwao\\b': ' wao ', '\\bnhay\\b': ' nhây ', '\\bz\\b': ' vậy ', '\\bcopy\\b': ' sao chép ', '\\bmake\\b': ' làm ở ',
                    '\\bng\\b': ' người ', '\\bsayy goodbye\\b': ' chào tạm biệt ', '\\bbcs\\b': ' bao cao su ', '\\bsag\\b': ' sang ', '\\bad\\b': ' quản trị viên ',
                    '\\b3s\\b': ' ba giây ', '\\bctay\\b': ' chia tay ', '\\bbo ̣ n na ̀ y cho đi tu ̀ hê ́ tao\\b': ' bọn này cho đi tù hết ',
                    '\\bcap\\b': ' chụp ', '\\bngheng\\b': ' nghen ', '\\bs mà\\b': ' sao mà ', '\\bt\\b': ' tao ', '\\bxún\\b': ' xuống ', '\\bmọe\\b': ' mẹ ',
                    '\\bkphai\\b': ' không phải ', '\\bshare\\b': ' chia sẻ ', '\\blist fr\\b': ' danh sách bạn bè ', '\\btụt mood\\b': ' tụt hứng ', '\\b10m\\b': ' mười mét ',
                    '\\brồu\\b': ' rồi ', '\\badmin\\b': ' quản trị viên ', '\\bnghi nguồn\\b': ' ghi nguồn ', '\\bmini\\b': ' nhỏ ', '\\bmớii\\b': ' mới ', '\\bng ta\\b': ' người ta ',
                    '\\bcsgt\\b': ' cảnh sát giao thông ', '\\biemmm\\b': ' em ', '\\bfucklong\\b': ' đm ', '\\bex\\b': ' người yêu cũ ', '\\bphìm\\b': ' phim ', '\\bchạp\\b': ' tập ',
                    '\\be\\b': ' em ', '\\bz\\b': ' vậy ', '\\bmóa\\b': ' má ', '\\baamir khan\\b': ' per ', '\\bnao\\b': ' nào ', '\\bnghia het\\b': ' nghĩa hết ',
                    '\\brốt cục\\b': ' rốt cuộc ', '\\bhỏg\\b': ' hỏng ', '\\buwu\\b': ' dễ thương ', '\\bbanh kem\\b': ' bánh kem ', '\\bsn\\b': ' sinh nhật ', '\\bdamege\\b': ' sát thương ',
                    '\\bsì phố\\b': ' thành phố ', '\\bcau view\\b': ' câu lượt xem ', '\\bonline\\b': ' trược tuyến ', '\\bchg\\b': ' chưa ', '\\bbb\\b': ' bạn bè ',
                    '\\bbổg\\b': ' bổng ', '\\bkaraok\\b': ' karaoke ', '\\bhic hic\\b': ' hu hu ', '\\bcái mẹt\\b': ' cái mặt ', '\\bthoy\\b': ' thôi ',
                    '\\bweeee\\b': ' hi hi ', '\\b:d\\b': ' ha ha ', '\\bcontent\\b': ' nội dung ', '\\bfree\\b': ' miễn phí ', '\\bcmt\\b': ' bình luận ',
                    '\\bhihee\\b': ' hi hi ', '\\blink\\b': 'đường dẫn', '\\bkq\\b': 'kết quả', "lol" : "lồn", "dume" : "đm",
                    '\\b thă ́ c mă ́ c ta ̣ i sao thă ̀ người da đen no ́ quen được con nho ̉ da tră ́ người ngon va ̃ i thê ́ nhy ̃ \\b': ' tao đang thắc mắc tại sao thằng người da đen nó quen được con nhỏ da trắng người ngon vãi thế nhỉ ',
                    '\\bcia\\b': ' kia ', '\\blếuuuuuu\\b': ' lếu ', '\\bpug\\b': ' orther ', '\\bđag\\b': ' đang ', '\\bvãi_chưởng\\b': ' vãi chưỡng ',
                    '\\bteam\\b': ' đội ', '\\b150trieu\\b': ' 150 triệu ', '\\bcàrôt\\b': ' cà rốt ', '\\ball\\b': ' tất cả ', '\\bxink\\b': ' xinh ',
                    '\\bcaooooo\\b': ' cao ', '\\bsong rồi\\b': ' xong rồi ', '\\bvalungtung\\b': ' va lung tung ', '\\bb\\b': ' bạn ', '\\bhk\\b': ' hông ',
                    '\\bxh\\b': ' xã hội ', '\\bnhưg\\b': ' nhưng ', '\\bđmmmmm\\b': ' đm ', '\\b:\\(\\b': ' nagative ', '\\bmlz\\b': ' mặt lồn vậy ',
                    '\\bmatlon\\b': ' mặt lồn ', '\\bdoctor\\b': ' bác sĩ ', '\\bgood\\b': ' tốt ', '\\bvừa vv and vv\\b': '', '\\bvv\\b': ' vân vân ',
                    '\\bbđ\\b': ' bê đê ', '\\bbthg\\b': ' bình thường ', '\\blở có\\b': ' lỡ có ', '\\blương nv\\b': ' lương nhân viên ', '\\bdt\\b': ' điện thoại ',
                    '\\bvliz\\b': ' vl ', '\\bml\\b': ' mặt lồn ', '\\bmiss you\\b': ' nhớ bạn ', '\\bv trờiiiiii\\b': ' vậy trời ', '\\bdzậy\\b': ' gì vậy ',
                    '\\bak\\b': ' à ', '\\bnhưng vãn\\b': ' nhưng vẫn ', '\\bgd\\b': ' giáo dục ', '\\boto\\b': ' ô tô ', '\\bng ta\\b': ' người ta ',
                    '\\bròii\\b': ' rồi ', '\\bbg\\b': ' bây giờ ', '\\bdhs\\b': '', '\\bstyle\\b': ' kiểu ', '\\bgvcn\\b': ' giáo viên chũ nhiệm ',
                    '\\bơiii\\b': ' ơi ', '\\bs\\b': ' sao ', '\\b17t\\b': ' 17 tuổi ', '\\bmạn cóa\\b': ' mạn quá ', '\\bhqua\\b': ' hôm qua ', '\\bnhưg\\b': ' nhưng ',
                    '\\bmv\\b': ' video ca nhạc ', '\\bđg\\b': ' đường ', '\\b0 ₫\\b': ' 0 điểm ', '\\bkill\\b': ' giết ', '\\bwtf\\b': ' what the fuck ',
                    '\\bhêt cả\\b': ' hết cả ', '\\bevery where\\b': ' mọi nơi ', '\\bdhs\\b': "đéo hiểu sao",  "\\bhmm\\b": "", "\\bhmmmmmmm\\b" : ""
    }
    for old_word, new_word in replacements.items():
        text = re.sub(old_word, new_word, text)

    # Thay thế dấu chấm thành dấu cách, giữ lại các biểu tượng cảm xúc
    text = re.sub(r'\.', ' ', text)

    return text

def replace_emoticons(text):
    #Quy các icon về 2 loại emoj: positive hoặc nagative
    emoticon_mapping = {
        "😠": "bực mình", "😂": "ha ha ", "😡" : "bực mình ", "🤧": "bực mình ", "🤤" : "hi hi ", "😈": "bực mình ", "😒": "đm ",
        "😭": "hu hu ", "😔" : "hic", "🙃": "🙂", "🤬" : "bực mình", "< 3" : "<3","❤️" : "<3", "❤" : "<3", "🤣" : "ha ha", "🌧": "mưa",
        "😏": "🙂", "🥰" : "<3 ", "💕": "<3 ", "😍": "<3 ", "💪": "cố lên ",  "☹️" : "hic", "🤢": "kinh tởm", "🤮" :"kinh tởm",
        "😆": "ha ha", "💓" : "<3", "😁": "ha ha ", "😊" : "hi hi ", "♥" : "<3", "💔" : "buồn", "😿" : "hu hu ", "😢" :"hu hu ",
        "😞": "hic ", "☺️" :"<3", "😘" : "<3",  "♥" : "<3", "🤑" : "hi hi", "😝": "hi hi", "😹": "ha ha", "😎": "ha ha",
        "🤩": "<3", "😻": "<3", "😱": "sợ vãi", "😄": "ha ha ", "😫": "hu hu ", "🤔": "? ",


    #Quy các ký icon về ký hiệu
        "😑" : "-_-" , "._." : "-_-", "-.-" : "-_-", ":|": "-_-", "😀": ":D", "☹" : "-_-",

    }
    # Biểu thức chính quy để tìm kiếm tất cả các emoticon trong văn bản
    emoticon_pattern = re.compile('|'.join(re.escape(k) for k in emoticon_mapping.keys()))

    # Hàm thay thế cho mỗi emoticon
    def replace(match):
        return emoticon_mapping[match.group(0)]

    # Thay thế các emoticon trong văn bản
    text = emoticon_pattern.sub(replace, text)

    # Biểu thức chính quy để thay thế nhiều dấu "(" hoặc ")" liên tiếp bằng một emoticon duy nhất
    consecutive_parentheses_pattern = re.compile(r'(\(+)|(\)+)')

    # Hàm thay thế cho mỗi chuỗi dấu "(" hoặc ")"
    def replace_consecutive_parentheses(match):
        if match.group(1):
            return '😭'
        elif match.group(2):
            return '😂'


    # Thay thế nhiều dấu "(" hoặc ")" liên tiếp bằng một emoticon duy nhất
    text = consecutive_parentheses_pattern.sub(replace_consecutive_parentheses, text)
    if "😂" in text:
        text = text.replace("😂", "ha ha")
    elif "😭" in text:
        text = text.replace("😭", "hu hu")
    # Danh sách các biểu tượng cần xóa
    icons_to_remove = ["🐶", "💋", "🕸", "🌸", "🙎🏻‍♀️", "💃", "🕸","🌧"," 🏻 ", "🤦🏻‍♀️" ,"🤨","😩","😶","😩","🙏","🤕", "😤","😌","💩",
                       "😕", "🙄", "🥵", "🤪", "🙄", "👌", "☝️", "😖","🤦🏻‍♂️", "🤦‍♀️", "😪", "☻", "🐶", "🐕", "🤦🏻‍♀️", "👻", "💋",
                       "👏", "🙏", "👏", "👏🏻", "🤔", "🤗", "😬", "🧐", "😅", "😭", "#", "🙁", "- -", "-_-", "😥", "🌝", "🌚", "😶",
                       "😥", "😪","😣", "🤭", "😳","🙁","🤯","😪", "😦", "😚", "😄", "🙁", "😪", "😥", "🤫", "🤕", "😮", "😃", "😜",
                       "😉","😐","😗","😴"
                      ]
    # Xóa các biểu tượng cần loại bỏ
    for icon in icons_to_remove:
        text = text.replace(icon, "")


    return text



# Định nghĩa lớp SentimentClassifier
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        x = self.drop(output)
        x = self.fc(x)
        return x

# Định nghĩa lớp SentimentPredictor
class SentimentPredictor:
    def __init__(self, model_path, class_names):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.model = self.load_model(model_path)
        self.class_names = class_names

    def load_model(self, model_path):
        model = SentimentClassifier(n_classes=7)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        model.eval()
        return model

    def predict_sentiment(self, text):
        encoded_review = self.tokenizer.encode_plus(
            text,
            max_length=50,  # Thay đổi theo độ dài tối đa bạn đã chọn khi huấn luyện mô hình
            truncation=True,
            add_special_tokens=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(device)
        attention_mask = encoded_review['attention_mask'].to(device)

        output = self.model(input_ids, attention_mask)
        _, y_pred = torch.max(output, dim=1)

        return self.class_names[y_pred]

# Sử dụng
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = r'D:\A.I\FSOFT_semester_4\Project\Group_1\Train_(7)\phobert_fold4.pth'
class_names = ["Enjoyment", "Disgust", "Sadness", "Anger", "Surprise", "Fear", "Other"]
predictor = SentimentPredictor(model_path, class_names)

#Nhập văn bản bạn muốn dự đoán
text_to_predict = input("Nhập văn bản cần dự đoán cảm xúc: ")
text_to_predict = replace_emoticons(replace_words(text_to_predict))
predicted_sentiment = predictor.predict_sentiment(text_to_predict)
print(f'Text: {text_to_predict}')
print(f'Predicted Sentiment: {predicted_sentiment}')

