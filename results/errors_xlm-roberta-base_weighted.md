# Səhv təhlili: xlm-roberta-base_weighted

Test: 7786, səhv: 1254 (16.1%)

## Dilimlər üzrə performans

| slice                             |    n |   accuracy |   macro_f1 |   F1_neg |   F1_neu |
|:----------------------------------|-----:|-----------:|-----------:|---------:|---------:|
| hamısı                            | 7786 |      0.839 |      0.601 |    0.752 |    0.146 |
| 1-2 söz                           | 1699 |      0.913 |      0.531 |    0.592 |    0.045 |
| 3-5 söz                           | 2308 |      0.886 |      0.601 |    0.729 |    0.135 |
| 6-15 söz                          | 2687 |      0.793 |      0.589 |    0.756 |    0.148 |
| 16+ söz                           | 1092 |      0.738 |      0.584 |    0.806 |    0.173 |
| Az. hərfləri var                  | 5024 |      0.830 |      0.604 |    0.756 |    0.154 |
| Az. hərfləri yox                  | 2762 |      0.855 |      0.593 |    0.745 |    0.120 |
| ziddiyyətli label (dublikatlarda) |   77 |      0.766 |      0.490 |    0.634 |    0.000 |

## Ən çox qarışdırılan cütlər (həqiqi → proqnoz)

| true     | pred     |   say |
|:---------|:---------|------:|
| positive | negative |   479 |
| negative | positive |   272 |
| positive | neutral  |   173 |
| neutral  | positive |   123 |
| neutral  | negative |   112 |
| negative | neutral  |    95 |

### negative → neutral (ulduz göstərilib)

- [2★] Z 445s4
- [1★] 2 ci dəfədir ki yaşayıram bu hadisəni bəzən çox gözəl vaxtında gəlir bəzən saatlarca sürücü gözləyirik yenə eyni hadisə oldu və zəhmət olmasa bu işə baxılsın sürücülər vaxtında təyin olunsun
- [1★] Cox vaxt yollari qatisdirir bir birine.
- [1★] Maaş köçəndə sms necə gəlir?
- [2★] Barmaq izini mütəmadi olaraq tanımır Tez-tez xətalar baş verir.
- [1★] Salam son yenilemeden sonra kart elave et bolmesinde tesdiqleme ucun uz seklini isteyir ama kamera acilmir yenilemezden evvel isleyirdi ama
- [1★] Nömrəni və şifrəni yazıram. Yazılır ki xəta baş verdi. Qeydiyyatdan keçmək olmur
- [1★] Belədə.məsləhət görmürəm
- [1★] Salam. Yeni versiya nə vaxt olacaqdır ? Məncə köhnə versiyası bundan daha rahat və yaxşı idi. Nəsə bu versiyanız məni bir o qədər də qane etmir.
- [2★] Bəzi əşyalar yoxdu açılmır
- [2★] Salam mən ƏDV vurmaq istəyirəm .yazılır ki gözlənilməyən xəta baş verdi nə edim??
- [1★] Proqramın istifadəsine gore elave pul kontur tutulur yoxsa meqabayt çıxılır?

### negative → positive (ulduz göstərilib)

- [1★] Ela piradramdi ele ancaq isdifade etmemisem
- [1★] Salam bunda internetdi güclü elemek olur
- [2★] Bonus verir bu proqram ödəniş edəndə.?
- [1★] ƏDV vurmaq olur?
- [1★] Salam xahiş edirəm cavab yazın .çatdırılma bölgələrini genişləndirib bakıda S Bəhluzadə küçəsi bülbülə qəsəbəsinə də sifariş qəbul edə bilərsiz?Və birdə əlavə etməliyəmki mən neçə dəfə məhlənin yanınd
- [1★] Arada bawi gedr
- [1★] Redd olsun
- [1★] Profilaktik işlərdən sonra ƏDV-lər silindi,geri qaytarmaq mümkündür?
- [1★] Tədbiq necə istifadə olunur?
- [2★] etrafli maraglan mag chin tel non toczy sizin
- [2★] ƏDV ÇEKİNİ NİYƏ VURMUR?
- [1★] 5 ulduz o vaxt olarsız ki təqaüd və pensiyaları vaxtında balansa köçürərsiz.

### neutral → negative (ulduz göstərilib)

- [3★] Yukləmişəm qeydiyyatdan keçmır axi koməklik edin
- [3★] Çox bərbad işləyir
- [3★] Men qeydiyyatin 3 cu seviyyesinde kod yaziram cavablandirmilar xahis edirem duzelis edin
- [3★] Salam carx legv olunub ?
- [3★] Bakıda Koroğlu metro stansiyasının yaxınlığında giriş qadağandır nişanı olduğu halda dönməyi göstərdi ( istiqamət Mərdəkan marşurutu dayanacağı )
- [3★] Xaiş edirəm artıq Uwalki düzəldin düz əməlli işləmir
- [3★] Cahsback məndə görünmür niyə?
- [3★] kontur yukləyə bilmirəm 31 mart və 1 aprel yoxladım alınmadı nəsə problem var
- [3★] Program Donur. telefon yeni 12 ram
- [3★] Promo kodu yalnız ödəniş kartı əlavə etdikdən sonra istifadə etmək olur (((
- [3★] Yeni guncelleme geldi pis gune Dusdu esasda şeher axtaranda düzeltseniz sevinerem
- [3★] Birbankın dizaynı dəyişib, digital kartın məlumatlarını tapa bilmədim. Güncəlləmə yaxşı şeydir, ancaq kökündən etməyin... Əsas menyu yerində qalsın ki,hər güncəlləmə zamanı yenidən proqram öyrənməyək.

### neutral → positive (ulduz göstərilib)

- [3★] Arada kəsilir adamı əsəb edir.amma yaxşıdı
- [3★] sevmədiyim bir şeyi varki deyilərki internetsiz işliyir amma səf məlumat veriler işləmir internetsiz
- [3★] Blede...
- [3★] Fotomaqnit 10eded 10azn. Sifaris: insta: kinder_az 👈🏻
- [3★] Hələki 3 kifayətdi
- [3★] Salam mənim ədv geri al kartımda pul var idi amma kart silindi ve necə geri ala bilərəm zəhmət olmasa cavablandırın
- [3★] I
- [3★] Tetbiq icerisinde bir nece profille qeydiyyatdan kecmek olsa dahada yaxsi olar
- [3★] Salam ubankda olan kartlarimin icindeki vesaitleru gore bilmirem umumiyyetle kartim eks olunur sadece boz ekram cixir zehmet olmasa bilen varsa desin duzeldim
- [3★] Samsung galaxsi s6 batarekasi var 2 dene razilasma yolu ile satiram
- [3★] Çatdirilma cox bahadi ama yaxsidi
- [3★] Sms bankçılıq xidməti ləğv edilib ?

### positive → negative (ulduz göstərilib)

- [4★] Bizim ünvanı Google mapla eyni göstərmir, digər proqramlar düzgün göstərir, amma sizdə küçənin nömrəsini başqa cür göstərir, bu halla son vaxtlar rastlaşmışam, təəssüf ki, məsafədə də çox böyük fərq e
- [5★] Xidmet yaxşıdı tətbiq problemdir çox sərtdir
- [5★] Amma kart mənim deyil atamindi mən nömrəyə yazdim ad soyad ata adi istədi versəm atama zəng edməzlər
- [5★] Karta daxil ola bilmirem
- [5★] Mənə görə bolt Bahadır. Digər şirkət var onu işlədirəm. 5 ulduzu da mesaj görünsün deyə verdim. Qardaş bahaçılıqdı
- [5★] Pitbull cins it axtardım o çıxmadı
- [5★] Köhnə elanı necə silmək olar?
- [5★] Bir kopiyliy bank deyil Cox seviyesiz bankdi
- [5★] Niye acilmir proqram?
- [5★] 28 % öz millətine çoxdu xarici vətəndaşlara olar. Amma özümüzə ən azından 15-20% olmalıdır biz axı Azərbaycanlıyıq
- [4★] Salam. Hədiyyə çarxında qeydiyyatdan keçmək olmur, ad və soyadı yazanda bir hərif yazan kimi klavyatura donur, digər hərifləri yazmaq olmur, niyə?
- [5★] Bezi peyserler dedelerinin qiymetine verir masini.. Allah insafinizi kessin... hec olmasa baha satirsansa cammati aldatmayin temiz masin satin... Allah qazandiginiz o haram pulu burnunuzdan getirsin i

### positive → neutral (ulduz göstərilib)

- [5★] Masini satmisan basqa masin almisam burda amma kohneni silib yeni aldigmi qeyd eliye bilmirem ikinci aftomaabili secmek olur o haldada aylig iki manat odemeyim lazim olur yeni birinci satdigim aftomab
- [4★] Yaxşı proqramdır amma kart əlavə etmək istəyirəm olmur artıq 1 aya yaxındı her gün yoxlayirsam hərgundə əlavə etmək olmur . Bu problemi aradan qaldırın zəhmət olmazsa
- [5★] Son zamanlar niyə işləmir ?Səbəbini bilmək olar?
- [4★] Pis demek olmaz. Normal
- [5★] Bu interneti ucuzlaşdirsaz yaxşı olar
- [5★] məlumat verə bilərsiz proqram nə vaxt düzələcək? bilirik ki 14:00dan bəri işləmir.
- [5★] Borda giriş elemk üçün ne edirler zehmetdi deye bilersiz
- [5★] Yenilesez lap yaxsi olar.
- [5★] Başqa nömrəyə çarxı necə fırlatmaq olur
- [4★] Biraz narahatlıqları var, kaşki daha operativ işləsəniz
- [5★] salam pulun keçdiyi göstərilir ancaq kartda göstərmir necə olur?
- [4★] Bir bankda dosta köçürmə edilmir, onlayn alış verişdə edə bilmirəm, nə etməliyəm?

## Hər iki model (xlm-roberta-base_weighted, tfidf_logreg) eyni cür yanılır: 798 nümunə

Bunların bir hissəsi label səhvidir (rəy mətni ilə ulduz uyğun gəlmir). Əl ilə yoxlayıb faizini hesabla: bu, modelin 'tavanı' haqqında məlumat verir.

- label=positive [4★], hər ikisi=neutral: 4 ulduz çünki bəzən açılmır😕
- label=neutral [3★], hər ikisi=negative: Proqram işləmir qeydiyyatdan kecmək isdəyirəm deyir müvəqqəti problem var nəzarət edin
- label=positive [5★], hər ikisi=negative: Salam teze yuklemisem bu proqram kontur yuklemek istyrem alinmir sebeb nedi
- label=positive [5★], hər ikisi=negative: Qeydiyyatdan kecmek olmur nomremi qeyd edirem kod gelmir nece aca bilerm
- label=negative [1★], hər ikisi=positive: 9l
- label=positive [5★], hər ikisi=negative: Proqram islemir. Cokub sisteminiz
- label=positive [5★], hər ikisi=neutral: Yəni pis deyil
- label=positive [5★], hər ikisi=negative: Salam yükləmişəm qeydiyyatdan keçə bilmirəm xaiş edirəm kömək edəsiniz
- label=positive [5★], hər ikisi=neutral: Hər şey yaxşıdı,bircə ƏDV düzgün işləmir.ƏDV-ləyığılıb qalır.
- label=negative [1★], hər ikisi=positive: AD N 527118
- label=positive [5★], hər ikisi=negative: mende ninja teklif gelmirr son zamanlar bu ne ile baglidi
- label=positive [5★], hər ikisi=negative: Telefonda NFC dəstəkləmir bu proqram
- label=positive [5★], hər ikisi=neutral: 😶
- label=negative [1★], hər ikisi=positive: Super apk
- label=negative [2★], hər ikisi=neutral: Salam ,Tetbiqde NFC funksiyasi itib.tetbiqi silib yaziram amma qayitmir NFC.Birde bu tetbiqi kim fikirleshib ve kim tesdiq edibse ona "ateshli" salamlar. Ve elece de 3 d secure de oz nomremi tesdiqley
- label=neutral [3★], hər ikisi=positive: Фввм Ф
- label=positive [4★], hər ikisi=neutral: Yenilənmələrə ehtiyac var
- label=positive [5★], hər ikisi=neutral: Torba S paketi bitenden sonra limitsiz olurmu??ses flan gedirmi vatcapda wekil zad.yoxsa tekce yaziwmadi.reyimize cvb versez sevinerdik.
- label=negative [2★], hər ikisi=positive: Hamı endirdim
- label=positive [4★], hər ikisi=negative: Məndə açılmır
- label=neutral [3★], hər ikisi=negative: hər dəfə şifrə istəyir bu yorucudur, vaxt itkisidir
- label=positive [5★], hər ikisi=negative: Qaranlıq rejim getirin zəhmət olmasa, istifadə etmək olmur onsuz
- label=neutral [3★], hər ikisi=positive: Çox gözəl praqlam di 1banq obri banklara baxaraq bu daha üstün du çətin olsada yaxşıdı
- label=positive [5★], hər ikisi=neutral: hərdən axmaq sürücüləri nəzərə almasaq pis deyil
- label=neutral [3★], hər ikisi=negative: Salam. Mən M10-n hesabıma pul göndərmişəm. İndi m10-na daxil olduqda məndən üz ilə təstiqlənmə istəyir. Deyir yenidən cəhd edin hər cür variantla etmişəm xahiş edirəm bu problemi həll edim, pul hesabı
- label=positive [5★], hər ikisi=negative: Serverde xeta niye yazir qeydiyyat etmek olmur.komeklik edin
- label=negative [1★], hər ikisi=positive: YAXŞI 123
- label=positive [5★], hər ikisi=negative: işlemir praqram
- label=neutral [3★], hər ikisi=positive: Kredit bölməsidə açıq olsa lap əla olar
- label=negative [1★], hər ikisi=neutral: Salam. Bu tetbiqe daxil olarken chetinlikle uzleshirem, demek olar barmaq izi ile daxil ola bilmirem ve ya, Her defe parolumu daxil edende yalnishdir yazir ve eyni qaydada davam edir. Zehmet olmasa ko