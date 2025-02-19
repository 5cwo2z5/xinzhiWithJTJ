URO_old = (
    (163, 156, 124),
    (171, 127, 73),
    (169, 97, 40),
    (162, 81, 35),
    (136, 50, 25)
)
URO = (
    (133, 126, 97),
    (151, 97, 46),
    (149, 67, 13),
    (142, 51, 8),
    (116, 20, 0)
)

BIL_old = (
    (169, 163, 126),
    (161, 126, 89),
    (164, 117, 85),
    (149, 87, 58)
)
BIL = (
    (156, 148, 108),
    (148, 111, 71),
    (151, 100, 67),
    (138, 70, 40)
)

KET_old = (
    (157, 134, 98),
    (127, 77, 72),
    (119, 62, 60),
    (90, 33, 38)
)
KET = (
    (159, 124, 87),
    (129, 67, 61),
    (121, 52, 49),
    (92, 23, 27)
)

CRE_old = (
    (164, 147, 82),
    (136, 123, 85),
    (120, 104, 77),
    (112, 97, 71),
    (108, 89, 66)
)
CRE = (
    (180, 154, 77),
    (152, 130, 80),
    (136, 111, 72),
    (128, 104, 66),
    (124, 96, 61)
)

BLO_old = (
    (175, 105, 3),
    (151, 96, 2),
    (96, 82, 18),
    (80, 77, 13)
)
BLO = (
    (179, 104, 0),
    (155, 95, 0),
    (100, 81, 15),
    (84, 76, 10)
)

PRO_old = (
    (167, 142, 71),
    (138, 124, 44),
    (119, 116, 48),
    (100, 106, 56),
    (89, 101, 60),
    (76, 93, 70)
)
PRO = (
    (167, 142, 40),
    (138, 124, 44),
    (119, 116, 48),
    (100, 106, 56),
    (89, 101, 60),
    (76, 93, 70)
)

MCA_old = (
    (162, 148, 123),
    (148, 152, 121),
    (132, 147, 120),
    (119, 142, 118)
)
MCA = (
    (158, 140, 117),
    (140, 140, 115),
    (128, 139, 114),
    (115, 134, 112)
)

NIT_old = (
    (149, 134, 106),
    (135, 118, 110),
    (131, 100, 106)
)
NIT = (
    (147, 131, 107),
    (133, 115, 111),
    (129, 97, 107)
)

LEU_old = (
    (158, 156, 127),
    (132, 137, 134),
    (120, 116, 118),
    (114, 105, 110),
    (96, 92, 102)
)
LEU = (
    (155, 144, 125),
    (129, 125, 132),
    (117, 104, 116),
    (111, 93, 108),
    (93, 80, 100)
)

GLU_old = (
    (160, 140, 25),
    (112, 122, 54),
    (82, 109, 60),
    (80, 107, 60),
    (70, 99, 65),
    (32, 76, 56)
)
GLU = (
    (168, 142, 42),
    (120, 124, 71),
    (90, 111, 77),
    (88, 109, 77),
    (78, 101, 82),
    (40, 78, 73)
)

SG_old  = (
    (50, 72, 41),
    (69, 82, 37),
    (75, 82, 32),
    (81, 84, 27),
    (91, 89, 31),
    (93, 89, 26),
    (102, 95, 30)
)
SG  = (
    (74, 87, 58),
    (93, 97, 54),
    (99, 97, 49),
    (105, 99, 44),
    (116, 104, 48),
    (117, 104, 43),
    (126, 110, 47)
)

PH_old  = (
    (127, 83, 51),
    (132, 101, 44),
    (80, 99, 40),
    (34, 81, 53),
    (19, 77, 62)
)
PH  = (
    (145, 103, 56),
    (142, 107, 53),
    (116, 115, 57),
    (80, 96, 55),
    (74, 93, 51)
)

VC_old = (
    (17, 78, 63),
    (28, 81, 82),
    (73, 104, 60),
    (107, 120, 60),
    (118, 124, 65)
)
VC = (
    (19, 63, 69),
    (30, 66, 69),
    (75, 89, 66),
    (109, 105, 66),
    (120, 109, 71)
)

CA_old  = (
    (158, 152, 129),
    (139, 136, 127),
    (133, 126, 131),
    (118, 96, 128)
)
CA  = (
    (137, 129, 109),
    (118, 113, 107),
    (112, 103, 111),
    (97, 73, 108)
)

standrad_data = {
    # 'CON': (),
    'URO': URO,
    'BIL': BIL,
    'KET': KET,
    'CRE': CRE,
    'BLO': BLO,
    'PRO': PRO,
    'MCA': MCA,
    'NIT': NIT,
    'LEU': LEU,
    'GLU': GLU,
    'SG': SG,
    'PH' : PH,
    'VC': VC,
    'CA': CA
}

rst_lst = {
    'URO': ('-', '-', '-', '++', '+++'),
    'BLO': ('-', '-', '-', '+'),
    'BIL': ('-', '-', '+', '++'),
    'KET': ('-', '-', '+', '++'),
    'CA': ('-', '-', '+', '++'),
    'LEU': ('-', '-', '-', '-', '微量'),
    'GLU': ('-', '-', '+', '++', '+++', '++++'),
    'PRO': ('-', '-', '-', '+', '++', '+++'),
    'PH': ('5', '5', '5', '6', '7'),
    'CRE': ('-', '-', '-', '++', '+++'),
    'NIT': ('-', '-', '弱阳性'),
    'SG': ('1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000'),
    'VC': ('0', '0', '0', '0', '0'),
    'MCA': ('-', '-', '-', '++')
}

rst_lst_bak = {
    'URO': ('-', '+', '++', '+++', '++++'),
    'BLO': ('-', '+', '++', '+++'),
    'BIL': ('-', '+', '++', '+++'),
    'KET': ('-', '+', '++', '+++'),
    'CA': ('-', '+', '++', '+++'),
    'LEU': ('-', '微量', '少量', '中量', '大量'),
    'GLU': ('-', '+-', '+', '++', '+++', '++++'),
    'PRO': ('-', '+-', '+', '++', '+++', '++++'),
    'PH': ('5', '6', '7', '8', '9'),
    'CRE': ('-', '+-', '+', '++', '+++'),
    'NIT': ('-', '弱阳性', '强阳性'),
    'SG': ('1.000', '1.005', '1.010', '1.015', '1.020', '1.025', '1.030'),
    'VC': ('0', '0.6', '1.4', '2.8', '5.6'),
    'MCA': ('-', '+', '++', '+++')
}
