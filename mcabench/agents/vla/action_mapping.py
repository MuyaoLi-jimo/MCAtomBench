'''
Author: Muyao 2350076251@qq.com
Date: 2025-02-18 15:57:29
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-30 21:41:47
'''

import re
import numpy as np
import copy
import pickle
import json
from collections import OrderedDict
from typing import Union, List, Dict
import torch
from tqdm import tqdm
from pathlib import Path
from rich import console
from abc import ABC, abstractmethod

from minestudio.utils.vpt_lib.actions import ActionTransformer, Buttons
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.simulator.entry import CameraConfig


def get_special_token(model_id: str, bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 11, 11]) -> list:
    """
    Generate a list of all unknown tokens to mark unknown tokens.

    Args:
        model_id (str): Model identifier used to load the corresponding tokenizer.
        bases (list): List of bases for buttons and camera.

    Returns:
        list: A list containing all unknown tokens.

    Note:
        It is assumed that the number 8641 will never appear.
    """
    from transformers import AutoTokenizer
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Calculate the required extra tokens based on the sum of bases plus an additional 30 tokens
    token_num = sum(bases) + 30
    # Sort and extract the last token_num special tokens from the vocabulary
    special_tokens = sorted(list(tokenizer.vocab.items()), key=lambda x: x[-1])[-token_num:]
    return special_tokens


def prepare_for_remap_control_token(tokenizer_type: str,
                                    bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21],
                                    not_text=True) -> dict:
    """
    Prepare a dictionary for remapping control tokens.

    Args:
        tokenizer_type (str): Type of tokenizer.
        bases (list): List of bases for each action group.
        not_text (bool): Flag to indicate whether to return non-text tokens.

    Returns:
        dict: A dictionary where keys are tokens and values are corresponding (group index, number) tuples.
    """
    tokens = {}
    # Iterate over each action group
    for i, base in enumerate(bases):
        for j in range(base):
            token = map_control_token(j, i, tokenizer_type, not_text=not_text)
            tokens[token] = (i, j)
    return tokens


def map_control_token(num: int, place: int, tokenizer_type: str, not_text: bool = False) -> str:
    """
    Map a number at a specific position to a control token.

    Args:
        num (int): The current number (index within the action group).
        place (int): The action group index.
        tokenizer_type (str): Tokenizer type;.
        not_text (bool): Determines whether to return text or a numerical identifier.

    Returns:
        str: The corresponding control token string.

    Raises:
        ValueError: If the specified tokenizer type is not supported.
    """
    if tokenizer_type == "qwen2_vl":
        # Define the list of special tokens, organized by action groups and indices
        special_tokens = [
            # Group 1: hotbar
            [["<|reserved_special_token_180|>", 151837],
             ["<|reserved_special_token_181|>", 151838],
             ["<|reserved_special_token_182|>", 151839],
             ["<|reserved_special_token_183|>", 151840],
             ["<|reserved_special_token_184|>", 151841],
             ["<|reserved_special_token_185|>", 151842],
             ["<|reserved_special_token_186|>", 151843],
             ["<|reserved_special_token_187|>", 151844],
             ["<|reserved_special_token_188|>", 151845],
             ["<|reserved_special_token_189|>", 151846]],
            # Group 2: 3 tokens "forward", "back”
            [["<|reserved_special_token_190|>", 151847],
             ["<|reserved_special_token_191|>", 151848],
             ["<|reserved_special_token_192|>", 151849]],
            # Group 3: 3 tokens "left", "right"
            [["<|reserved_special_token_193|>", 151850],
             ["<|reserved_special_token_194|>", 151851],
             ["<|reserved_special_token_195|>", 151852]],
            # Group 4: 3 tokens, "sprint" "sneak"
            [["<|reserved_special_token_196|>", 151853],
             ["<|reserved_special_token_197|>", 151854],
             ["<|reserved_special_token_198|>", 151855]],
            # Group 5: 2 tokens, representing "use"
            [["<|reserved_special_token_199|>", 151856],
             ["<|reserved_special_token_200|>", 151857]],
            # Group 6: 2 tokens, representing "drop"
            [["<|reserved_special_token_201|>", 151858],
             ["<|reserved_special_token_202|>", 151859]],
            # Group 7: 2 tokens, representing "attack"
            [["<|reserved_special_token_203|>", 151860],
             ["<|reserved_special_token_204|>", 151861]],
            # Group 8: 2 tokens, representing "jump"
            [["<|reserved_special_token_205|>", 151862],
             ["<|reserved_special_token_206|>", 151863]],
            # Group 9: 2 tokens, representing "camera"
            [["<|reserved_special_token_207|>", 151864],
             ["<|reserved_special_token_208|>", 151865]],
            # Group 10: 2 tokens, representing "inventory"
            [["<|reserved_special_token_176|>", 151833],
             ["<|reserved_special_token_177|>", 151834]],
            # Group 11: camera
            [["<|reserved_special_token_209|>", 151866],
             ["<|reserved_special_token_210|>", 151867],
             ["<|reserved_special_token_211|>", 151868],
             ["<|reserved_special_token_212|>", 151869],
             ["<|reserved_special_token_213|>", 151870],
             ["<|reserved_special_token_214|>", 151871],
             ["<|reserved_special_token_215|>", 151872],
             ["<|reserved_special_token_216|>", 151873],
             ["<|reserved_special_token_217|>", 151874],
             ["<|reserved_special_token_218|>", 151875],
             ["<|reserved_special_token_219|>", 151876],
             ["<|reserved_special_token_220|>", 151877],
             ["<|reserved_special_token_221|>", 151878],
             ["<|reserved_special_token_222|>", 151879],
             ["<|reserved_special_token_223|>", 151880],
             ["<|reserved_special_token_224|>", 151881],
             ["<|reserved_special_token_225|>", 151882],
             ["<|reserved_special_token_226|>", 151883],
             ["<|reserved_special_token_227|>", 151884],
             ["<|reserved_special_token_228|>", 151885],
             ["<|reserved_special_token_229|>", 151886]],
            # Group 12: camera
            [["<|reserved_special_token_230|>", 151887],
             ["<|reserved_special_token_231|>", 151888],
             ["<|reserved_special_token_232|>", 151889],
             ["<|reserved_special_token_233|>", 151890],
             ["<|reserved_special_token_234|>", 151891],
             ["<|reserved_special_token_235|>", 151892],
             ["<|reserved_special_token_236|>", 151893],
             ["<|reserved_special_token_237|>", 151894],
             ["<|reserved_special_token_238|>", 151895],
             ["<|reserved_special_token_239|>", 151896],
             ["<|reserved_special_token_240|>", 151897],
             ["<|reserved_special_token_241|>", 151898],
             ["<|reserved_special_token_242|>", 151899],
             ["<|reserved_special_token_243|>", 151900],
             ["<|reserved_special_token_244|>", 151901],
             ["<|reserved_special_token_245|>", 151902],
             ["<|reserved_special_token_246|>", 151903],
             ["<|reserved_special_token_247|>", 151904],
             ["<|reserved_special_token_248|>", 151905],
             ["<|reserved_special_token_249|>", 151906],
             ["<|reserved_special_token_250|>", 151907]],
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    
    try:
        # Return either the text token or numeric identifier based on not_text flag (index 0 or 1)
        token = special_tokens[place][num][not_text]
    except Exception as e:
        print("place:", place, "num:", num, "not_text:", not_text, e)
    return token


def remap_control_token(token: str, use_num: bool = True, tokenizer_type: str = "qwen2_vl") -> tuple:
    """
    Map a control token back to its corresponding action information.

    Args:
        token (str): The control token.
        use_num (bool): Whether to use numeric mapping.
        tokenizer_type (str): Tokenizer type; currently supports only "qwen2_vl".

    Returns:
        tuple: (action group index, number) tuple. Returns (-1, -1) if token not found.
    """
    re_tokens = {}
    if tokenizer_type == "qwen2_vl":
        # Define the mapping dictionary from token to action (when use_num is True)
        if use_num:
            re_tokens = {
                151837: [0, 0], 151838: [0, 1], 151839: [0, 2], 151840: [0, 3], 151841: [0, 4],
                151842: [0, 5], 151843: [0, 6], 151844: [0, 7], 151845: [0, 8], 151846: [0, 9],
                151847: [1, 0], 151848: [1, 1], 151849: [1, 2],
                151850: [2, 0], 151851: [2, 1], 151852: [2, 2],
                151853: [3, 0], 151854: [3, 1], 151855: [3, 2],
                151856: [4, 0], 151857: [4, 1],
                151858: [5, 0], 151859: [5, 1],
                151860: [6, 0], 151861: [6, 1],
                151862: [7, 0], 151863: [7, 1],
                151864: [8, 0], 151865: [8, 1],
                151833: (9, 0), 151834: (9, 1),
                151866: [10, 0], 151867: [10, 1], 151868: [10, 2], 151869: [10, 3], 151870: [10, 4],
                151871: [10, 5], 151872: [10, 6], 151873: [10, 7], 151874: [10, 8], 151875: [10, 9],
                151876: [10, 10], 151877: [10, 11], 151878: [10, 12], 151879: [10, 13], 151880: [10, 14],
                151881: [10, 15], 151882: [10, 16], 151883: [10, 17], 151884: [10, 18], 151885: [10, 19],
                151886: [10, 20],
                151887: [11, 0], 151888: [11, 1], 151889: [11, 2], 151890: [11, 3],
                151891: [11, 4], 151892: [11, 5], 151893: [11, 6], 151894: [11, 7], 151895: [11, 8],
                151896: [11, 9], 151897: [11, 10], 151898: [11, 11], 151899: [11, 12], 151900: [11, 13],
                151901: [11, 14], 151902: [11, 15], 151903: [11, 16], 151904: [11, 17], 151905: [11, 18],
                151906: [11, 19], 151907: [11, 20],
            }
        else:
            raise ValueError(f"{tokenizer_type} can't use text as tokens")
    elif tokenizer_type == "llama_2":
        if use_num:
            re_tokens = {
                31536: (0, 0), 31537: (0, 1), 31563: (0, 2), 31571: (0, 3), 31578: (0, 4), 31582: (0, 5), 31585: (0, 6), 31598: (0, 7), 
                31603: (0, 8), 31604: (0, 9), 31000: (0, 10), 31001: (0, 11), 31002: (0, 12), 31003: (0, 13), 31004: (0, 14), 31005: (0, 15), 
                31006: (0, 16), 31007: (0, 17), 31008: (0, 18), 31009: (0, 19), 31010: (0, 20), 31011: (0, 21), 31012: (0, 22), 31013: (0, 23), 
                31014: (0, 24), 31015: (0, 25), 31016: (0, 26), 31017: (0, 27), 31018: (0, 28), 31019: (0, 29), 31020: (0, 30), 31021: (0, 31), 31022: (0, 32), 31023: (0, 33), 31024: (0, 34), 31025: (0, 35), 31026: (0, 36), 31027: (0, 37), 
                31028: (0, 38), 31029: (0, 39), 31030: (0, 40), 31031: (0, 41), 31032: (0, 42), 31033: (0, 43), 31034: (0, 44), 31035: (0, 45), 31036: (0, 46), 31037: (0, 47), 31038: (0, 48), 31039: (0, 49), 31040: (0, 50), 31041: (0, 51), 
                31042: (0, 52), 31043: (0, 53), 31044: (0, 54), 31045: (0, 55), 31046: (0, 56), 31047: (0, 57), 31048: (0, 58), 31049: (0, 59), 31050: (0, 60), 31051: (0, 61), 31052: (0, 62), 31053: (0, 63), 31054: (0, 64), 31055: (0, 65), 
                31056: (0, 66), 31057: (0, 67), 31058: (0, 68), 31059: (0, 69), 31060: (0, 70), 31061: (0, 71), 31062: (0, 72), 31063: (0, 73), 31064: (0, 74), 31065: (0, 75), 31066: (0, 76), 31067: (0, 77), 31068: (0, 78), 31069: (0, 79), 
                31070: (0, 80), 31071: (0, 81), 31072: (0, 82), 31073: (0, 83), 31074: (0, 84), 31075: (0, 85), 31076: (0, 86), 31077: (0, 87), 31078: (0, 88), 31079: (0, 89), 31080: (0, 90), 31081: (0, 91), 31082: (0, 92), 31083: (0, 93), 
                31084: (0, 94), 31085: (0, 95), 31086: (0, 96), 31087: (0, 97), 31088: (0, 98), 31089: (0, 99), 31090: (0, 100), 31091: (0, 101), 31092: (0, 102), 31093: (0, 103), 31094: (0, 104), 31095: (0, 105), 31096: (0, 106), 31097: (0, 107), 
                31098: (0, 108), 31099: (0, 109), 31100: (0, 110), 31101: (0, 111), 31102: (0, 112), 31103: (0, 113), 31104: (0, 114), 31105: (0, 115), 31106: (0, 116), 31107: (0, 117), 31108: (0, 118), 31109: (0, 119), 31110: (0, 120), 31111: (0, 121), 
                31112: (0, 122), 31113: (0, 123), 31114: (0, 124), 31115: (0, 125), 31116: (0, 126), 31117: (0, 127), 31118: (0, 128), 31119: (0, 129), 31120: (0, 130), 31121: (0, 131), 31122: (0, 132), 31123: (0, 133), 31124: (0, 134), 31125: (0, 135), 31126: (0, 136), 31127: (0, 137), 31128: (0, 138), 31129: (0, 139), 31130: (0, 140), 31131: (0, 141), 31132: (0, 142), 31133: (0, 143), 31134: (0, 144), 31135: (0, 145), 31136: (0, 146), 31137: (0, 147), 31138: (0, 148), 31139: (0, 149), 
                31140: (0, 150), 31141: (0, 151), 31142: (0, 152), 31143: (0, 153), 31144: (0, 154), 31145: (0, 155), 31146: (0, 156), 31147: (0, 157), 31148: (0, 158), 31149: (0, 159), 31150: (0, 160), 31151: (0, 161), 31152: (0, 162), 31153: (0, 163), 31154: (0, 164), 31155: (0, 165), 31156: (0, 166), 31157: (0, 167), 31158: (0, 168), 31159: (0, 169), 31160: (0, 170), 31161: (0, 171), 31162: (0, 172), 31163: (0, 173), 31164: (0, 174), 31165: (0, 175), 31166: (0, 176), 31167: (0, 177), 31168: (0, 178), 
                31169: (0, 179), 31170: (0, 180), 31171: (0, 181), 31172: (0, 182), 31173: (0, 183), 31174: (0, 184), 31175: (0, 185), 31176: (0, 186), 31177: (0, 187), 31178: (0, 188), 31179: (0, 189), 31180: (0, 190), 31181: (0, 191), 31182: (0, 192), 31183: (0, 193), 31184: (0, 194), 31185: (0, 195), 31186: (0, 196), 31187: (0, 197), 31188: (0, 198), 31189: (0, 199), 31190: (0, 200), 31191: (0, 201), 31192: (0, 202), 31193: (0, 203), 31195: (0, 204), 31196: (0, 205), 31197: (0, 206), 31198: (0, 207), 
                31199: (0, 208), 31200: (0, 209), 31201: (0, 210), 31202: (0, 211), 31203: (0, 212), 31204: (0, 213), 31205: (0, 214), 31206: (0, 215), 31207: (0, 216), 31208: (0, 217), 31209: (0, 218), 31210: (0, 219), 31211: (0, 220), 31212: (0, 221), 31213: (0, 222), 31214: (0, 223), 31215: (0, 224), 31216: (0, 225), 31217: (0, 226), 31218: (0, 227), 31219: (0, 228), 31220: (0, 229), 31221: (0, 230), 31222: (0, 231), 31223: (0, 232), 31224: (0, 233), 31225: (0, 234), 31226: (0, 235), 31227: (0, 236), 
                31228: (0, 237), 31229: (0, 238), 31230: (0, 239), 31231: (0, 240), 31232: (0, 241), 31233: (0, 242), 31234: (0, 243), 31235: (0, 244), 31236: (0, 245), 31237: (0, 246), 31238: (0, 247), 31239: (0, 248), 31240: (0, 249), 31241: (0, 250), 31243: (0, 251), 31244: (0, 252), 31245: (0, 253), 31246: (0, 254), 31247: (0, 255), 31248: (0, 256), 31249: (0, 257), 31250: (0, 258), 31251: (0, 259), 31252: (0, 260), 31253: (0, 261), 31254: (0, 262), 31255: (0, 263), 31256: (0, 264), 31257: (0, 265), 
                31258: (0, 266), 31259: (0, 267), 31260: (0, 268), 31261: (0, 269), 31262: (0, 270), 31263: (0, 271), 31264: (0, 272), 31265: (0, 273), 31266: (0, 274), 31267: (0, 275), 31268: (0, 276), 31269: (0, 277), 31270: (0, 278), 31271: (0, 279), 31272: (0, 280), 31273: (0, 281), 31274: (0, 282), 31275: (0, 283), 31276: (0, 284), 31277: (0, 285), 31278: (0, 286), 31279: (0, 287), 31280: (0, 288), 31281: (0, 289), 31282: (0, 290), 31283: (0, 291), 31284: (0, 292), 31285: (0, 293), 31286: (0, 294), 
                31287: (0, 295), 31288: (0, 296), 31289: (0, 297), 31290: (0, 298), 31291: (0, 299), 31292: (0, 300), 31293: (0, 301), 31294: (0, 302), 31295: (0, 303), 31296: (0, 304), 31297: (0, 305), 31298: (0, 306), 31299: (0, 307), 31300: (0, 308), 31301: (0, 309), 31302: (0, 310), 31303: (0, 311), 31304: (0, 312), 31305: (0, 313), 31306: (0, 314), 31307: (0, 315), 31308: (0, 316), 31309: (0, 317), 31310: (0, 318), 31311: (0, 319), 31312: (0, 320), 31313: (0, 321), 31314: (0, 322), 31315: (0, 323), 
                31316: (0, 324), 31317: (0, 325), 31318: (0, 326), 31319: (0, 327), 31320: (0, 328), 31321: (0, 329), 31322: (0, 330), 31323: (0, 331), 31324: (0, 332), 31325: (0, 333), 31326: (0, 334), 31327: (0, 335), 31328: (0, 336), 31329: (0, 337), 31330: (0, 338), 31331: (0, 339), 31332: (0, 340), 
                31333: (0, 341), 31334: (0, 342), 31335: (0, 343), 31336: (0, 344), 31337: (0, 345), 31338: (0, 346), 31339: (0, 347), 31340: (0, 348), 31341: (0, 349), 31342: (0, 350), 31343: (0, 351), 31344: (0, 352), 31345: (0, 353), 31346: (0, 354), 31347: (0, 355), 31348: (0, 356), 31349: (0, 357), 31350: (0, 358), 31351: (0, 359), 31352: (0, 360), 31353: (0, 361), 31354: (0, 362), 31355: (0, 363), 31356: (0, 364), 31357: (0, 365), 31358: (0, 366), 31359: (0, 367), 31360: (0, 368), 31361: (0, 369), 
                31362: (0, 370), 31363: (0, 371), 31364: (0, 372), 31365: (0, 373), 31366: (0, 374), 31367: (0, 375), 31368: (0, 376), 31369: (0, 377), 31370: (0, 378), 31371: (0, 379), 31372: (0, 380), 31373: (0, 381), 31374: (0, 382), 31375: (0, 383), 31376: (0, 384), 31377: (0, 385), 31378: (0, 386), 
                31380: (0, 387), 31381: (0, 388), 31382: (0, 389), 31383: (0, 390), 31384: (0, 391), 31385: (0, 392), 31386: (0, 393), 31387: (0, 394), 31388: (0, 395), 31389: (0, 396), 31390: (0, 397), 31391: (0, 398), 31392: (0, 399), 31393: (0, 400), 31394: (0, 401), 31395: (0, 402), 31396: (0, 403), 31397: (0, 404), 31398: (0, 405), 31399: (0, 406), 31400: (0, 407), 31401: (0, 408), 31402: (0, 409), 31403: (0, 410), 31404: (0, 411), 
                31405: (0, 412), 31406: (0, 413), 31407: (0, 414), 31408: (0, 415), 31409: (0, 416), 31410: (0, 417), 31411: (0, 418), 31412: (0, 419), 31413: (0, 420), 31414: (0, 421), 31415: (0, 422), 31416: (0, 423), 31417: (0, 424), 31418: (0, 425), 31419: (0, 426), 31420: (0, 427), 31421: (0, 428), 31422: (0, 429), 31423: (0, 430), 31424: (0, 431), 31425: (0, 432), 31426: (0, 433), 31427: (0, 434), 31428: (0, 435), 31429: (0, 436), 31430: (0, 437), 31431: (0, 438), 31432: (0, 439), 31433: (0, 440), 
                31434: (0, 441), 31435: (0, 442), 31436: (0, 443), 31437: (0, 444), 31438: (0, 445), 31439: (0, 446), 31440: (0, 447), 31441: (0, 448), 31442: (0, 449), 31443: (0, 450), 31444: (0, 451), 31445: (0, 452), 31446: (0, 453), 31447: (0, 454), 31448: (0, 455), 31449: (0, 456), 31450: (0, 457), 
                31451: (0, 458), 31452: (0, 459), 31453: (0, 460), 31454: (0, 461), 31455: (0, 462), 31456: (0, 463), 31457: (0, 464), 31458: (0, 465), 31459: (0, 466), 31460: (0, 467), 31461: (0, 468), 31462: (0, 469), 31463: (0, 470), 31464: (0, 471), 31465: (0, 472), 31466: (0, 473), 31467: (0, 474), 31468: (0, 475), 31469: (0, 476), 31470: (0, 477), 31471: (0, 478), 31472: (0, 479), 31473: (0, 480), 31474: (0, 481), 31475: (0, 482), 31476: (0, 483), 31477: (0, 484), 31478: (0, 485), 31479: (0, 486), 
                31480: (0, 487), 31481: (0, 488), 31482: (0, 489), 31483: (0, 490), 31484: (0, 491), 31485: (0, 492), 31486: (0, 493), 31487: (0, 494), 31488: (0, 495), 31489: (0, 496), 31490: (0, 497), 31491: (0, 498), 31492: (0, 499), 31493: (0, 500), 31494: (0, 501), 31495: (0, 502), 31496: (0, 503), 31497: (0, 504), 31498: (0, 505), 31499: (0, 506), 31500: (0, 507), 31501: (0, 508), 31502: (0, 509), 31503: (0, 510), 31504: (0, 511)
            }
        else:
            raise ValueError(f"{tokenizer_type} can't use text as tokens")
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    # Return (-1, -1) if token is not found
    return re_tokens.get(token, (-1, -1))


def tag_token(place: int, tokenizer_type: str, return_type: int = 0):
    """
    Return the start or end tag token based on the position.

    Args:
        place (int): 0 for the start tag, 1 for the end tag.
        tokenizer_type (str): Tokenizer type;.
        return_type (int): Specifies which part of the token to return: 0 for token text, 1 for numeric identifier.

    Returns:
        tuple: (token text, token numeric identifier)
    """
    assert place in {0, 1}
    if tokenizer_type == "qwen2_vl":
        special_tokens = [
            ('<|reserved_special_token_178|>', 151835),
            ('<|reserved_special_token_179|>', 151836),
        ]
    elif tokenizer_type == "llama-2":
        special_tokens = [('유', 31533),('요', 31527)]
    elif tokenizer_type=="llama-3":
        special_tokens = [('<|reserved_special_token_178|>', 128183), ('<|reserved_special_token_179|>', 128184),]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][return_type]


class ActionTokenizer(ABC):
    """
    Base class for action tokenizers, used to encode and decode actions to and from tokens.
    """
    # Define common movement and operation actions
    movements = ('forward', 'back', 'left', 'right', 'sprint', 'sneak')
    operations = ('use', 'drop', 'attack', 'jump')

    def __init__(self,
                 tokenizer_type="qwen2_vl",
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1,
                 camera_maxval=10):
        self.tokenizer_type = tokenizer_type

        # Retrieve the start and end tag tokens and their IDs
        self.act_beg_id = tag_token(0, self.tokenizer_type, return_type=1)
        self.act_end_id = tag_token(1, self.tokenizer_type, return_type=1)
        self.act_beg_token = tag_token(0, self.tokenizer_type, return_type=0)
        self.act_end_token = tag_token(1, self.tokenizer_type, return_type=0)

        # Initialize camera configuration
        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
        )
        self.n_camera_bins = camera_config.n_camera_bins

        # Define the null action with default values (False for buttons, (0.0, 0.0) for camera)
        self.null_action = {
            'forward': False, 'back': False, 'left': False, 'right': False,
            'sprint': False, 'sneak': False,
            'hotbar.1': False, 'hotbar.2': False, 'hotbar.3': False, 'hotbar.4': False,
            'hotbar.5': False, 'hotbar.6': False, 'hotbar.7': False, 'hotbar.8': False, 'hotbar.9': False,
            'use': False, 'drop': False, 'attack': False, 'jump': False,
            'inventory': False,
            'camera': (0.0, 0.0)
        }

        # Initialize action transformer and action mapper
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)

    @abstractmethod
    def encode(self, actions: Dict) -> Union[torch.Tensor, list, str]:
        """
        Abstract method: Encode actions into tokens.

        Args:
            actions (Dict): Dictionary of actions.

        Returns:
            Union[torch.Tensor, list, str]: Encoded token representation.
        """
        pass

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, list]) -> List[OrderedDict]:
        """
        Abstract method: Decode tokens into actions.

        Args:
            tokens (Union[torch.Tensor, list]): Sequence of tokens (string type is not allowed).

        Returns:
            List[OrderedDict]: A list of decoded actions as OrderedDict objects.
        """
        pass


class OneActionTokenizer(ActionTokenizer):
    """
    Single action tokenizer that implements the specific encoding and decoding logic.

    BUTTONS_GROUPS:
        Names of different action groups.
    """
    BUTTONS_GROUPS = [
        "hotbar", "fore or back", "left or right", "sprint or sneak", "use",
        "drop", "attack", "jump", "camera"
    ]

    def __init__(self,
                 tokenizer_type="llama-2",
                 bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21],
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1):
        # Call the parent constructor to initialize common configurations
        super().__init__(tokenizer_type=tokenizer_type,
                         camera_quantization_scheme=camera_quantization_scheme,
                         camera_mu=camera_mu,
                         camera_binsize=camera_binsize)
        # Log related information using rich console
        console.Console().log(f"tokenizer_type: {tokenizer_type}")
        console.Console().log(f"bases: {bases}, camera_mu: {camera_mu}, n_camera_bins: {self.n_camera_bins}, camera_binsize: {camera_binsize}")
        self.bases = bases
        # NULL_ACTION is the default null action; its encoding uses the middle values of the last two elements of bases
        self.NULL_ACTION = [0, (bases[-2] // 2) * bases[-2] + (bases[-1] // 2)]
    
    def decode(self,tokens:Union[torch.Tensor,List]):
        """decode the tokens to action
        """
        group_actions = self.token_2_group_action(tokens,)
        
        actions = [self.group_action_2_decimal_action(group_action) for group_action in group_actions ]
        action_dicts = []
        for action in  actions:
            action_dict = {
                "buttons":np.array([action[0]]),
                "camera":np.array([action[1]]),  #返回一个工作
            }
            action_dict = OrderedDict({key: value[0] for key, value in action_dict.items()})
            action_dicts.append(action_dict)
        return action_dicts

    def encode(self, trajectory: dict) -> list[tuple[int]]:
        """
        Encode the action trajectory into tokens.

        Args:
            trajectory (dict): Dictionary containing actions, observations, frame IDs, and UUIDs.

        Returns:
            list: A list of encoded trajectories, each containing control token, observations, UUID, and frame information.
        """
        minerl_actions = trajectory['actions']
        traj_len = len(minerl_actions['attack'])
        # Retrieve additional trajectory information (observations, frame IDs, UUIDs)
        observations = trajectory.get('observations', [""] * traj_len)
        frame_ids = trajectory.get('frame_ids', range(0, traj_len))
        uuids = trajectory.get('uuids', [""] * traj_len)

        # Convert action values for buttons and camera into numpy arrays
        minerl_action_transformed = {key: np.array(val)
                                     for key, val in minerl_actions.items()
                                     if key in Buttons.ALL or key == "camera"}
        # Convert environment actions to policy-friendly action format
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        # Convert to factorized action representation using the mapper
        actions = self.action_mapper.from_factored(minerl_action)

        action_list = []
        # Store each frame's action as a tuple (buttons, camera)
        for idx in range(traj_len):
            action_list.append((actions["buttons"][idx][0], actions["camera"][idx][0]))

        encoded_trajectory = []
        # Generate control tokens for each action and combine with additional information into a dictionary
        for idx, action in enumerate(action_list):
            control_token = self.encode_action(action)
            encoded_trajectory.append({
                "action_token": control_token,
                "observations": [observations[idx]],
                'uuid': uuids[idx],
                'frames': (frame_ids[idx], 1, frame_ids[idx]),
            })
        return encoded_trajectory

    def encode_action(self, action: tuple) -> str:
        """
        Encode a single action into a control token string.

        Args:
            action (tuple): A tuple (buttons, camera).

        Returns:
            str: The encoded control token string.
        """
        # Ensure the action has two parts
        assert len(action) == 2
        # Convert decimal action to group action representation
        group_action = self.decimal_action_2_group_action(action)
        # Convert group action representation to token string
        tokens = self.group_action_2_token(group_action)
        return tokens

    def group_action_2_token(self, group_action):
        """
        Convert a group action representation into a control token string.

        Args:
            group_action: A list of numbers representing each part of the action.

        Returns:
            str: The concatenated control token string (with start and end tags).
        """
        # Map each group number to its corresponding control token
        zero_include_token_list = [map_control_token(num, i, self.tokenizer_type)
                                     for i, num in enumerate(group_action)]
        # Concatenate tokens for non-zero actions (excluding the last 4 tokens; camera tokens are handled separately)
        control_token = ''.join((s for x, s in zip(group_action[:-4], zero_include_token_list[:-4]) if x != 0))
        # Append camera-related tokens (ensure camera action information is preserved)
        control_token = control_token + "".join((s for s in zero_include_token_list[-2:]))
        # Add start and end tag tokens around the control token
        tag_control_token = self.act_beg_token + control_token + self.act_end_token
        return tag_control_token

    def token_2_group_action(self, tokens: Union[torch.Tensor, list]):
        """
        Convert a token sequence into a group action representation.

        Args:
            tokens (Union[torch.Tensor, list]): Sequence of tokens representing actions.

        Returns:
            list: A list of group action representations (each as a list of numbers).
        """
        actions = []
        # Initialize a default group action with zeros; for camera parts, use the midpoint values
        action_base = [0] * len(self.bases)
        camera_null = [self.bases[-1] // 2, self.bases[-2] // 2]
        action_base[-2:] = camera_null

        # Convert torch.Tensor tokens to list if necessary
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 2:
                tokens = tokens.squeeze()
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            raise ValueError("wrong type!")

        start_idx = 0
        # Split the token sequence based on start and end tag tokens; each segment represents one action
        while start_idx < len(tokens):
            try:
                first_index_n1 = tokens.index(self.act_beg_id, start_idx)
                first_index_n2 = tokens.index(self.act_end_id, first_index_n1 + 1)
            except ValueError:
                break

            # Extract control tokens between the start and end tags
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
            action = copy.copy(action_base)
            # Map each control token back to its corresponding group number and update the action
            for token in control_tokens:
                place, num = remap_control_token(token, use_num=True, tokenizer_type=self.tokenizer_type)
                if place != -1:
                    action[place] = num

            # If camera part is not equal to the default, set the inventory flag (set the fourth-last element to 1)
            if action[-2:] != camera_null:
                action[-4] = 1

            actions.append(copy.copy(action))
            start_idx = first_index_n2 + 1

        # If no actions are parsed, return the default null action
        if len(actions) == 0:
            actions.append(action_base)

        return actions

    def decimal_action_2_group_action(self, inputs: tuple):
        """
        Convert a decimal action representation into a group action representation with varying bases.

        Args:
            inputs (tuple): A tuple of two decimal integers representing button and camera actions.

        Returns:
            tuple: Each element represents the value for one action group.

        Description:
            - For button actions, perform successive modulo and integer division operations according to the bases.
            - If the button part equals 8640, mark it as inventory mode and set it to 0.
            - For camera actions, process the last two parts separately.
        """
        decimals = list(inputs)
        result = [0] * len(self.bases)
        inventory_flag = False

        # Check if the button part is 8640; if so, enable the inventory flag and set it to 0
        if decimals[0] == 8640:
            inventory_flag = True
            decimals[0] = 0
        else:
            # Convert the button part from lower to higher digits
            for i in range(len(self.bases) - 4, -1, -1):
                result[i] = decimals[0] % self.bases[i]
                decimals[0] //= self.bases[i]

        # Process the camera part: first the last digit, then the second last
        result[-1] = decimals[1] % self.bases[-1]
        decimals[1] //= self.bases[-1]
        result[-2] = decimals[1] % self.bases[-2]
        decimals[1] //= self.bases[-2]

        # If inventory flag is True, set the third-last element to 1
        if inventory_flag:
            result[-3] = 1
        if decimals != [0, 0]:
            print(decimals)
            raise ValueError("The decimal number is too large for the custom base system.")
        return tuple(result)

    def group_action_2_decimal_action(self, inputs):
        """
        Convert a group action representation with varying bases into a decimal action representation.

        Args:
            inputs: A list of numbers, with the length matching the bases.

        Returns:
            tuple: The converted decimal action representation, including button and camera parts.

        Raises:
            ValueError: If the input length does not match the expected number of digits or exceeds base limits.
        """
        if len(inputs) != len(self.bases):
            raise ValueError("The input number does not match the expected number of digits.")
        decimal_results = [0, 0]
        mid = len(inputs) - 3  # Boundary between button and camera parts

        # Calculate the decimal value for the button part
        for i, digit in enumerate(inputs):
            if digit >= self.bases[i]:
                raise ValueError(f"Digit at position {i} exceeds the base limit of {self.bases[i]-1}.")
            if i < mid:
                decimal_results[0] = decimal_results[0] * self.bases[i] + digit
            elif i == mid and digit:
                decimal_results[0] = 8640  # Special inventory flag
            else:
                decimal_results[1] = decimal_results[1] * self.bases[i] + digit
        return tuple(decimal_results)

    def null_token(self) -> str:
        """
        Get the token corresponding to the null action.

        Returns:
            str: The control token string for the null action.
        """
        return self.encode_action(self.NULL_ACTION)
    