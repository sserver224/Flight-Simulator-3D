import re
import os
import sys
if os.name!='nt':
        try:
                from tkinter import messagebox
                tkinter_available=True
        except:
                tkinter_available=False
        if tkinter_available:
                if sys.platform.startswith('darwin'):
                        messagebox.showerror('Incompatible OS', 'This game is Windows only, you are currently on a Mac system.')
                else:
                        messagebox.showerror('Incompatible OS', 'This game is Windows only, you are currently on a Linux system.')
        else:        
                if sys.platform.startswith('darwin'):
                        command='display dialog "This game is Windows only, you are currently on a Mac system. This program will now close." buttons {"OK"} default button 1'
                        os.system("osascript -e '"+command+"'")
                else:
                        print('This game is Windows only, you are currently on a Linux system.')
                        input('Press ENTER to continue...')
        sys.exit(1)
import time
import json
import math
import struct
import socket
import threading
from math import *
import subprocess
from os import path
from tkinter.messagebox import *
errorlist=[]
try:
        import harfang as hg
except ImportError:
        errorlist.append('harfang')
from random import *
try:
        from XInput import *
except ImportError:
        errorlist.append('XInput-python')
except IOError:
        errorlist.append('XInput_fail')
from os import getcwd
from tkinter import *
from random import random
from tkinter.ttk import *
from random import uniform
from tkinter import messagebox
if len(errorlist)>0:
        if 'harfang' in errorlist:
                if 'XInput_fail' in errorlist:
                        messagebox.showerror('Critical Error', 'Module harfang is missing or damaged. Reinstall with:\npip install harfang\nXInput failed to load.\nThis program will now close.')
                elif 'XInput-python' in errorlist:
                        messagebox.showerror('Critical Error', 'Module harfang is missing or damaged. Reinstall with:\npip install harfang\nModule XInput is missing or damaged.\nReinstall with:\npip install XInput-python\nThis program will now close.')
                else:
                        messagebox.showerror('Critical Error', 'Module harfang is missing or damaged. Reinstall with:\npip install harfang\nThis program will now close.')
                sys.exit(1)
        else:
                if 'XInput_fail' in errorlist:
                        messagebox.showwarning('Warning', 'XInput failed to load. Controller support will be disabled.')
                elif 'XInput-python' in errorlist:
                        messagebox.showwarning('Warning', 'Module XInput is missing or damaged. Reinstall with:\npip install XInput-python\nController support will be disabled.')
                def get_connected():
                        return (False, False, False, False)
def get_local_file_path(relative_path):
        try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS
        except Exception:
                base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
def conform_string(s):
	# Remove all non-word characters (everything except numbers and letters)
	s = re.sub(r"[^\w\s]", '', s)
	# Replace all runs of whitespace with a single dash
	s = re.sub(r"\s+", '_', s)
	return s

def list_to_color(c: list):
	return hg.Color(c[0], c[1], c[2], c[3])


def color_to_list(c: hg.Color):
	return [c.r, c.g, c.b, c.a]


def list_to_vec2(v: list):
	return hg.Vec2(v[0], v[1])


def vec2_to_list(v: hg.Vec2):
	return [v.x, v.y]


def list_to_vec3(v: list):
	return hg.Vec3(v[0], v[1], v[2])


def list_to_vec3_radians(v: list):
	v = list_to_vec3(v)
	v.x = radians(v.x)
	v.y = radians(v.y)
	v.z = radians(v.z)
	return v

def list_to_mat4(v: list):
	return hg.TransformationMat4(hg.Vec3(v[0], v[1], v[2]), hg.Vec3(v[3], v[4], v[5]), hg.Vec3(v[6], v[7], v[8]))


def mat4_to_list(v: hg.Mat4):
	p, r, s = hg.Decompose(v)
	return [p.x, p.y, p.z, r.x, r.y, r.z, s.x, s.y, s.z]

def vec3_to_list(v: hg.Vec3):
	return [v.x, v.y, v.z]


def vec3_to_list_degrees(v: hg.Vec3):
	l = vec3_to_list(v)
	l[0] = degrees(l[0])
	l[1] = degrees(l[1])
	l[2] = degrees(l[2])
	return l


def load_json_matrix(file_name):
	file = hg.OpenText(file_name)
	json_script = hg.ReadString(file)
	hg.Close(file)
	if json_script != "":
		script_parameters = json.loads(json_script)
		pos = list_to_vec3(script_parameters["position"])
		rot = list_to_vec3_radians(script_parameters["rotation"])
		return pos, rot
	return None, None


def save_json_matrix(pos: hg.Vec3, rot: hg.Vec3, output_filename):
	script_parameters = {"position": vec3_to_list(pos), "rotation": vec3_to_list_degrees(rot)}
	json_script = json.dumps(script_parameters, indent=4)
	file = hg.OpenWrite(output_filename)
	hg.WriteString(file, json_script)
	hg.Close(file)
hostname = socket.gethostname()
HOST = socket.gethostbyname(hostname)

message_buffer = {}
send_message_thread = None
condition = threading.Condition()
logger = ""
sock = 0


def broadcast_msg(msg, port):
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
	s.sendto(msg, ("<broadcast>", port))


def connect_socket(c_host, port):
	global sock

	# Create a socket (SOCK_STREAM means a TCP socket)
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	# Connect to server and send data
	while (1):
		try:
			return sock.connect((c_host, port)) == 0
		except:
			pass


def listener_socket(port_):
	global sock, logger
	if sock != 0:
		sock.close()
	# Create a socket (SOCK_STREAM means a TCP socket)
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_socket.bind((HOST, port_))
	server_socket.listen(1)
	sock, address = server_socket.accept()
	logger = "{} connected".format(address)


def close_socket():
	sock.close()


def check_send_message(cv):
	while 1:
		with cv:
			cv.wait()
			m = message_buffer
			# create header with id of the binding function and the length of the message
			values = (m["id"], len(m["m"]))
			packer = struct.Struct('hi')
			packed_data = packer.pack(*values)

			sock.sendall(packed_data + m["m"])


# send only the last message in the thread
def send_message_quick(id, message):
	global send_message_thread, message_buffer
	with condition:
		message_buffer = {"id": id, "m": message}
		condition.notifyAll()

	if send_message_thread is None:
		send_message_thread = threading.Thread(target=check_send_message, args=(condition,))
		send_message_thread.start()


def send_message_with_id(id, message):
	# create header with id of the binding function and the length of the message
	values = (id, len(message))
	packer = struct.Struct('hi')
	packed_data = packer.pack(*values)

	sock.sendall(packed_data + message)


def send_message(message):
	# create header with id of the binding function and the length of the message
	size = len(message)
	sizeb = size.to_bytes(4, byteorder='big')
	sock.sendall(sizeb + message)

	"""
	values = (len(message),)
	packer = struct.Struct('i')
	packed_data = packer.pack(*values)

	sock.sendall(packed_data + message)
	"""


def get_answer_header_with_id():
	received = sock.recv(8)
	if len(received) <= 0:
		return None
	state = struct.unpack('hi', received)
	return state[1]  # size of the waiting message


def get_answer_header():  # int with the length of the msg
	global logger
	try:
		received = sock.recv(4)
		while len(received) > 0 and len(received) < 4:
			received += sock.recv(4 - len(received))

		if len(received) <= 0:
			return None
		size = int.from_bytes(received, "big")
		return size  # size of the waiting message
	except Exception:
		logger = "Error: Crash socket get_answer_header\n {0}".format(sys.exc_info()[0])
		return None


def get_answer(with_id=False, max_size_before_flush=-1):
	global logger
	try:
		if with_id:
			size = get_answer_header_with_id()
		else:
			size = get_answer_header()
		if size is None or (max_size_before_flush != -1 and size > max_size_before_flush):
			return None

		received = sock.recv(size)

		while len(received) < size:
			received += sock.recv(size - len(received))

		return received
	except Exception:
		logger = "Error: Crash socket get answer\n {0}".format(sys.exc_info()[0])
		return None
def get_hierarchy_strate(parent: dict, nodes: hg.NodeList):  # parent={"node":parent_node,"children":[]}
    for i in range(nodes.size()):
        f = False
        nd = nodes.at(i)
        if parent["node"] is None:
            if nd.GetTransform().GetParent() is None:
                f = True
        elif nd.GetTransform().GetParent().GetUid() == parent["node"].GetUid():
            f = True
        if f:
            child = {"node": nd, "children": []}
            get_hierarchy_strate(child, nodes)
            parent["children"].append(child)


def duplicate_hierarchy_to_NodeList(scene: hg.Scene, parent_node: hg.Node, children: list, dupl: hg.NodeList, suffix):
    for i, nd_dict in enumerate(children):
        nd = nd_dict["node"]
        new_node = duplicate_node_object(scene, nd, nd.GetName() + suffix)
        if parent_node is None:
            pn = "None"
        else:
            pn = parent_node.GetName()
        # print("new node : " + new_node.GetName() + " - Parent: " + pn)
        if parent_node is not None: new_node.GetTransform().SetParent(parent_node)
        dupl.push_back(new_node)
        duplicate_hierarchy_to_NodeList(scene, new_node, nd_dict["children"], dupl, suffix)


def duplicate_node_and_children(scene: hg.Scene, parent_node: hg.Node, nodes: hg.NodeList, suffix):
    hierarchy = {"node": parent_node, "children": []}
    """
    for i in range(nodes.size()):
        nd= nodes.at(i)
        if nd.GetTransform().GetParent() is None:
            pn="None"
        else: pn=nd.GetTransform().GetParent().GetName()
        print("Node list - NAME: "+nodes.at(i).GetName() + " - Parent: "+pn)
    """
    get_hierarchy_strate(hierarchy, nodes)
    # print("Nombre de nodes roots : " + str(len(hierarchy["children"])))
    dupl = hg.NodeList()
    if parent_node is not None:
        new_parent_node = duplicate_node_object(scene, parent_node, parent_node.GetName() + suffix)
        dupl.push_back(new_parent_node)
    else:
        new_parent_node = None
    duplicate_hierarchy_to_NodeList(scene, new_parent_node, hierarchy["children"], dupl, suffix)
    return dupl


def duplicate_node_object(scene: hg.Scene, original_node: hg.Node, name):
    ot = original_node.GetTransform()
    obj = original_node.GetObject()
    if obj is None:
        # print("Original node: "+original_node.GetName() + " NO OBJECT")
        new_node = scene.CreateNode(name)
        tr = scene.CreateTransform(ot.GetPos(), ot.GetRot(), ot.GetScale())
        new_node.SetTransform(tr)
    else:
        # print("Original node: " + original_node.GetName() + " OBJECT OK")
        mdl_ref = obj.GetModelRef()
        n = obj.GetMaterialCount()
        materials = []
        for k in range(n):
            materials.append(obj.GetMaterial(k))
        new_node = hg.CreateObject(scene, hg.TransformationMat4(ot.GetPos(), ot.GetRot(), ot.GetScale()), mdl_ref, materials)
        new_node.SetName(name)
    return new_node


def get_node_in_list(name, ndlist: hg.NodeList):
    for i in range(ndlist.size()):
        # print(ndlist.at(i).GetName())
        if ndlist.at(i).GetName() == name:
            return ndlist.at(i)
    return None


def create_spatialized_sound_state(loop):
    state = hg.SpatializedSourceState(hg.TranslationMat4(hg.Vec3(0, 0, 0)))
    state.volume = 1
    state.repeat = loop
    return state


def create_stereo_sound_state(loop):
    state = hg.StereoSourceState()
    state.volume = 1
    state.repeat = loop
    return state


def play_stereo_sound(stereo_ref, stereo_state):
    stereo_state[0].panning = -1
    stereo_state[1].panning = 1
    return [hg.PlayStereo(stereo_ref[0], stereo_state[0]), hg.PlayStereo(stereo_ref[1], stereo_state[1])]


def set_stereo_volume(stereo_ref, volume):
    hg.SetSourceVolume(stereo_ref[0], volume)
    hg.SetSourceVolume(stereo_ref[1], volume)


def get_pixel_bilinear(picture: hg.Picture, pos: hg.Vec2):
    w = picture.GetWidth()
    h = picture.GetHeight()
    if w==0: w=1
    if h==0: h=1
    x = (pos.x * w - 0.5) % w
    y = (pos.y * h - 0.5) % h
    xi = int(x)
    yi = int(y)
    xf = x - xi
    yf = y - yi
    xi1 = (xi + 1) % w
    yi1 = (yi + 1) % h
    c1 = picture.GetPixelRGBA(xi, yi)
    c2 = picture.GetPixelRGBA(xi1, yi)
    c3 = picture.GetPixelRGBA(xi, yi1)
    c4 = picture.GetPixelRGBA(xi1, yi1)
    c12 = c1 * (1 - xf) + c2 * xf
    c34 = c3 * (1 - xf) + c4 * xf
    c = c12 * (1 - yf) + c34 * yf
    return c


class MathsSupp:
	@classmethod
	def smoothstep(cls, edge0, edge1, x):
		t = max(0, min((x - edge0) / (edge1 - edge0), 1.0))
		return t * t * (3.0 - 2.0 * t)

	@classmethod
	def rotate_vector(cls, point: hg.Vec3, axe: hg.Vec3, angle):
		axe = hg.Normalize(axe)
		dot_prod = point.x * axe.x + point.y * axe.y + point.z * axe.z
		cos_angle = cos(angle)
		sin_angle = sin(angle)

		return hg.Vec3(
			cos_angle * point.x + sin_angle * (axe.y * point.z - axe.z * point.y) + (1 - cos_angle) * dot_prod * axe.x, \
			cos_angle * point.y + sin_angle * (axe.z * point.x - axe.x * point.z) + (1 - cos_angle) * dot_prod * axe.y, \
			cos_angle * point.z + sin_angle * (axe.x * point.y - axe.y * point.x) + (1 - cos_angle) * dot_prod * axe.z)

	@classmethod
	def rotate_matrix(cls, mat, axe: hg.Vec3, angle):
		axeX = hg.GetX(mat)
		axeY = hg.GetY(mat)
		# axeZ=hg.GetZ(mat)
		axeXr = cls.rotate_vector(axeX, axe, angle)
		axeYr = cls.rotate_vector(axeY, axe, angle)
		axeZr = hg.Cross(axeXr, axeYr)  # cls.rotate_vector(axeZ,axe,angle)
		return hg.Mat3(axeXr, axeYr, axeZr)

	@classmethod
	def rotate_vector_2D(cls, p: hg.Vec2, angle):
		cos_angle = cos(angle)
		sin_angle = sin(angle)

		return hg.Vec2(p.x * cos_angle - p.y * sin_angle, p.x * sin_angle + p.y * cos_angle)

	@classmethod
	def get_sound_distance_level(cls, sounder_view_position: hg.Vec3):
		distance = hg.Len(sounder_view_position)
		return 1 / (distance / 10 + 1)

	@classmethod
	def get_mix_color_value(cls, f, colors):
		if f < 1:
			fc = f * (len(colors) - 1)
			i = int(fc)
			fc -= i
			return colors[i] * (1 - fc) + colors[i + 1] * fc
		else:
			return colors[-1]
class Temporal_Perlin_Noise:
	def __init__(self, interval=0.1):
		self.pt_prec = 0
		self.b0 = 0
		self.b1 = 0
		self.date = 0
		self.interval = interval

	def temporal_Perlin_noise(self, dts):
		self.date += dts
		t = self.date / self.interval
		pr = int(t)
		t -= self.pt_prec

		if pr > self.pt_prec:
			self.pt_prec = pr
			self.b0 = self.b1
			self.b1 = uniform(-1, 1)
			t = 0

		return self.b0 + (self.b1 - self.b0) * (sin(t * pi - pi / 2) * 0.5 + 0.5)


# ============================================================================================
#			Intersection box / ray
# ============================================================================================

def precompute_ray(dir: hg.Vec3):
	try:
		x = 1 / dir.x
	except:
		x = inf

	try:
		y = 1 / dir.y
	except:
		y = inf

	try:
		z = 1 / dir.z
	except:
		z = inf

	invdir = hg.Vec3(x, y, z)
	sign = [invdir.x < 0, invdir.y < 0, invdir.z < 0]
	return invdir, sign


def compute_relative_ray(pos, dir, mat):
	mati = hg.InverseFast(mat)
	rotmat = hg.GetRotationMatrix(mati)
	return mati * pos, rotmat * dir


def intersect_box_ray(bounds, pos, dir, invdir, sign, dist):
	# bounds=[mm.mn,mm.mx]
	tmin = (bounds[sign[0]].x - pos.x) * invdir.x
	tmax = (bounds[1 - sign[0]].x - pos.x) * invdir.x
	tymin = (bounds[sign[1]].y - pos.y) * invdir.y
	tymax = (bounds[1 - sign[1]].y - pos.y) * invdir.y

	if (tmin > tymax) or (tymin > tmax):
		return False
	if tymin > tmin:
		tmin = tymin
	if tymax < tmax:
		tmax = tymax

	tzmin = (bounds[sign[2]].z - pos.z) * invdir.z
	tzmax = (bounds[1 - sign[2]].z - pos.z) * invdir.z

	if (tmin > tzmax) or (tzmin > tmax):
		return False

	if tzmin > tmin:
		tmin = tzmin
	if tzmax < tmax:
		tmax = tzmax

	impact = pos + dir * tmin

	if hg.Len(impact - pos) > dist: return False

	return True
class Overlays:
	# ================= Lines 3D
	vtx_decl_lines = None
	lines_program = None
	lines = []

	# ================= Texts 3D
	font_program = None
	debug_font = None
	text_matrx = None
	text_uniform_set_values = hg.UniformSetValueList()
	text_uniform_set_texture_list = hg.UniformSetTextureList()
	text_render_state = None
	texts3D_display_list = []
	texts2D_display_list = []

	@classmethod
	def init(cls):
		cls.vtx_decl_lines = hg.VertexLayout()
		cls.vtx_decl_lines.Begin()
		cls.vtx_decl_lines.Add(hg.A_Position, 3, hg.AT_Float)
		cls.vtx_decl_lines.Add(hg.A_Color0, 3, hg.AT_Float)
		cls.vtx_decl_lines.End()
		cls.lines_program = hg.LoadProgramFromAssets("shaders/pos_rgb")

		cls.font_program = hg.LoadProgramFromAssets("core/shader/font.vsb", "core/shader/font.fsb")
		cls.debug_font = hg.LoadFontFromAssets("font/default.ttf", 64)
		cls.text_matrx = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(0), hg.Deg(0), hg.Deg(0)), hg.Vec3(1, -1, 1))
		cls.text_uniform_set_values.push_back(hg.MakeUniformSetValue("u_color", hg.Vec4(1, 1, 0, 1)))
		cls.text_render_state = hg.ComputeRenderState(hg.BM_Alpha, hg.DT_Disabled, hg.FC_Disabled)

	@classmethod
	def display_named_vector(cls, position, direction, label, label_offset2D, color, label_size=0.012):
		if label != "":
			cls.add_text2D_from_3D_position(label, position, label_offset2D, label_size, color)
		cls.add_line(position, position + direction, color, color)

	@classmethod
	def display_vector(cls, position, direction, color0=hg.Color.Yellow, color1=hg.Color.Orange):
		cls.lines.append([position, position + direction, color0, color1])

	@classmethod
	def display_boxe(cls, vertices, color):
		links = [0, 1,
				 1, 2,
				 2, 3,
				 3, 0,
				 5, 6,
				 6, 7,
				 7, 4,
				 4, 5,
				 0, 4,
				 1, 5,
				 2, 6,
				 3, 7]
		for i in range(len(links) // 2):
			cls.lines.append([vertices[links[i * 2]], vertices[links[i * 2 + 1]], color, color])

	@classmethod
	def add_line(cls, p0, p1, c0, c1):
		cls.lines.append([p0, p1, c0, c1])

	@classmethod
	def draw_lines(cls, vid):
		vtx = hg.Vertices(cls.vtx_decl_lines, len(cls.lines) * 2)
		for i, line in enumerate(cls.lines):
			vtx.Begin(i * 2).SetPos(line[0]).SetColor0(line[2]).End()
			vtx.Begin(i * 2 + 1).SetPos(line[1]).SetColor0(line[3]).End()
		hg.DrawLines(vid, vtx, cls.lines_program)

	@classmethod
	def display_physics_debug(cls, vid, physics):
		physics.RenderCollision(vid, cls.vtx_decl_lines, cls.lines_program, hg.ComputeRenderState(hg.BM_Opaque, hg.DT_Disabled, hg.FC_Disabled), 1)

	@classmethod
	def get_2d(cls, camera, point3d: hg.Vec3, resolution: hg.Vec2):
		cam_mat = camera.GetTransform().GetWorld()
		view_matrix = hg.InverseFast(cam_mat)
		c = camera.GetCamera()
		projection_matrix = hg.ComputePerspectiveProjectionMatrix(c.GetZNear(), c.GetZFar(), hg.FovToZoomFactor(c.GetFov()), hg.Vec2(resolution.x / resolution.y, 1))
		pos_view = view_matrix * point3d
		f, pos2d = hg.ProjectToScreenSpace(projection_matrix, pos_view, resolution)
		if f:
			return hg.Vec2(pos2d.x, pos2d.y) / resolution
		else:
			return None

	@classmethod
	def get_2d_vr(cls, vr_hud_pos: hg.Vec3, point3d: hg.Vec3, resolution: hg.Vec2, head_matrix: hg.Mat4, z_near, z_far):
		fov = atan(vr_hud_pos.y / (2 * vr_hud_pos.z)) * 2
		vs = hg.ComputePerspectiveViewState(head_matrix, fov, z_near, z_far, hg.Vec2(vr_hud_pos.x / vr_hud_pos.y, 1))
		pos_view = vs.view * point3d
		f, pos2d = hg.ProjectToScreenSpace(vs.proj, pos_view, resolution)
		if f:
			return hg.Vec2(pos2d.x, pos2d.y)
		else:
			return None

	@classmethod
	def add_text3D(cls, text, pos, size, color, h_align=hg.DTHA_Left):
		cls.texts3D_display_list.append({"text": text, "pos": pos, "size": size, "color": color, "h_align": h_align, "font": cls.debug_font})

	@classmethod
	def display_texts3D(cls, vid, camera_matrix):
		for txt in cls.texts3D_display_list:
			cls.display_text3D(vid, camera_matrix, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"], txt["h_align"])

	@classmethod
	def display_text3D(cls, vid, camera_matrix, text, pos, size, font, color, h_align=hg.DTHA_Center):

		"""
		cam_pos = hg.GetT(cam_mat)
		az = hg.Normalize(pos-cam_pos)
		ax = hg.Cross(hg.Vec3.Up, az)
		ay = hg.Cross(az, ax)
		mat = hg.Mat3(ax, ay, az)
		"""
		mat = hg.GetRotationMatrix(camera_matrix)
		cls.text_uniform_set_values.clear()
		cls.text_uniform_set_values.push_back(hg.MakeUniformSetValue("u_color", hg.Vec4(color.r, color.g, color.b, color.a)))  # Color
		hg.DrawText(vid, font, text, cls.font_program, "u_tex", 0,
					hg.TransformationMat4(pos, mat, hg.Vec3(1, -1, 1) * size),  # * (size * resolution.y / 64)),
					hg.Vec3(0, 0, 0),
					h_align, hg.DTVA_Bottom,
					cls.text_uniform_set_values, cls.text_uniform_set_texture_list, cls.text_render_state)

	@classmethod
	def add_text2D_from_3D_position(cls, text, pos3D, offset2D, size, color, font=None, h_align=hg.DTHA_Left):
		cls.add_text2D(text, pos3D, size, color, font, h_align, True, offset2D)

	@classmethod
	def add_text2D(cls, text, pos, size, color, font=None, h_align=hg.DTHA_Left, convert_to_2D=False, offset2D=None):
		if font is None:
			font = cls.debug_font
		cls.texts2D_display_list.append({"text": text, "pos": pos, "offset2D": offset2D, "size": size, "color": color, "h_align": h_align, "font": font, "convert_to_2D": convert_to_2D})

	@classmethod
	def display_texts2D(cls, vid, camera, resolution):
		for txt in cls.texts2D_display_list:
			pos = txt["pos"]
			if txt["convert_to_2D"]:
				pos = cls.get_2d(camera, pos, resolution)
				if pos is None:
					continue
				if "offset2D" in txt and txt["offset2D"] is not None:
					pos += txt["offset2D"]
			cls.display_text2D(vid, resolution, txt["text"], pos, txt["size"], txt["font"], txt["color"], txt["h_align"])

	@classmethod
	def display_text2D(cls, vid, resolution, text, pos, size, font, color, h_align):
		cls.text_uniform_set_values.clear()
		cls.text_uniform_set_values.push_back(hg.MakeUniformSetValue("u_color", hg.Vec4(color.r, color.g, color.b, color.a)))  # Color
		hg.DrawText(vid, font, text, cls.font_program, "u_tex", 0,
					hg.TransformationMat4(hg.Vec3(pos.x * resolution.x, pos.y * resolution.y, 1), hg.Vec3(0, 0, 0), hg.Vec3(1, -1, 1) * (size * resolution.y / 64)),
					hg.Vec3(0, 0, 0),
					h_align, hg.DTVA_Bottom,
					cls.text_uniform_set_values, cls.text_uniform_set_texture_list, cls.text_render_state)

	@classmethod
	def display_texts2D_vr(cls, vid, head_matrix: hg.Mat4, z_near, z_far, resolution, vr_matrix, vr_hud_pos):
		for txt in cls.texts2D_display_list:
			pos = txt["pos"]
			if txt["convert_to_2D"]:
				pos = cls.get_2d_vr(vr_hud_pos, pos, resolution, head_matrix, z_near, z_far)
				if pos is None:
					continue
				if "offset2D" in txt and txt["offset2D"] is not None:
					pos += txt["offset2D"]
			cls.display_text2D_vr(vid, vr_matrix, vr_hud_pos, resolution, txt["text"], pos, txt["size"], txt["font"], txt["color"], txt["h_align"])

	@classmethod
	def display_text2D_vr(cls, v_id, vr_matrix, vr_hud_pos, resolution, text, pos, size, font, color, h_align):
		pos_vr = hg.Vec3((pos.x - 0.5) * vr_hud_pos.x, (pos.y - 0.5) * vr_hud_pos.y, vr_hud_pos.z - 0.01)
		scale2D = hg.Vec3(1, -1, 1) * (size * resolution.y / 64)
		scale_vr = hg.Vec3(scale2D.x / resolution.x * vr_hud_pos.x, scale2D.y / resolution.y * vr_hud_pos.y, 1)
		matrix = vr_matrix * hg.TransformationMat4(pos_vr, hg.Vec3(0, 0, 0), scale_vr)

		cls.text_uniform_set_values.clear()
		cls.text_uniform_set_values.push_back(hg.MakeUniformSetValue("u_color", hg.Vec4(color.r, color.g, color.b, color.a)))  # Color
		hg.DrawText(v_id, font, text, cls.font_program, "u_tex", 0,
					matrix,
					hg.Vec3(0, 0, 0),
					h_align, hg.DTVA_Bottom,
					cls.text_uniform_set_values, cls.text_uniform_set_texture_list, cls.text_render_state)
air_density0 = 1.225 #  sea level standard atmospheric pressure, 101325 Pa
F_gravity = hg.Vec3(0, -9.8, 0)

scene = None
scene_physics = None
water_level = 0


terrain_heightmap = None
terrain_position = hg.Vec3(-24896, -296.87, 9443)
terrain_scale = hg.Vec3(41480, 1000, 19587)
map_bounds = hg.Vec2(0, 255)


def init_physics(scn, scn_physics, terrain_heightmap_file, p_terrain_pos, p_terrain_scale, p_map_bounds):
	global scene, scene_physics, terrain_heightmap, terrain_position, terrain_scale, map_bounds
	scene = scn
	scene_physics = scn_physics
	terrain_heightmap = hg.Picture()
	terrain_position = p_terrain_pos
	terrain_scale = p_terrain_scale
	map_bounds = p_map_bounds
	hg.LoadPicture(terrain_heightmap, terrain_heightmap_file)


def get_terrain_altitude(pos: hg.Vec3):
	global terrain_position, terrain_scale, terrain_heightmap, map_bounds
	pos2 = hg.Vec2((pos.x - terrain_position.x) / terrain_scale.x, 1 - (pos.z - terrain_position.z) / terrain_scale.z)
	return get_map_altitude(pos2), get_terrain_normale(pos2)


def get_map_altitude(pos2d):
	global terrain_position, terrain_scale, terrain_heightmap, map_bounds
	a = (get_pixel_bilinear(terrain_heightmap, pos2d).r * 255 - map_bounds.x) / (map_bounds.y - map_bounds.x)
	a = max(water_level, a * terrain_scale.y + terrain_position.y)
	return a


def get_terrain_normale(pos2d):
	w = terrain_heightmap.GetWidth()
	h = terrain_heightmap.GetHeight()
	f = 1 / max(w, h)
	xd = hg.Vec2(f, 0)
	zd = hg.Vec2(0, f)
	return hg.Normalize(hg.Vec3(get_map_altitude(pos2d - xd) - get_map_altitude(pos2d + xd), 2 * f, get_map_altitude(pos2d - zd) - get_map_altitude(pos2d + zd)))


def _compute_atmosphere_temp(altitude):
	"""
	Internal function to compute atmospheric temperature according to altitude. Different layers have
	different temperature gradients, therefore the calculation is branched.
	Model is taken from ICAO DOC 7488: Manual of ICAO Standard Atmosphere.

	:param altitude: altitude in meters.
	:return: temperature in Kelvin.
	"""

	#Gradients are Kelvin/km.
	if altitude < 11e3:
		temperature_gradient = -6.5 #Kelvin per km.
		reference_temp = 288.15 #Temperature at sea level.
		altitude_diff = altitude - 0
	else:
		temperature_gradient = 0
		reference_temp = 216.65 #Temperature at 11km altitude.
		altitude_diff = altitude - 11e3

	return reference_temp + temperature_gradient*(altitude_diff / 1000)



def compute_atmosphere_density(altitude):
	# Barometric formula
	# temperature_K : based on ICAO Standard Atmosphere
	temperature_K = _compute_atmosphere_temp(altitude)
	R = 8.3144621  # ideal (universal) gas constant, 8.31446 J/(mol·K)
	M = 0.0289652  # molar mass of dry air, 0.0289652 kg/mol
	g = 9.80665  # earth-surface gravitational acceleration, 9.80665 m/s2
	d = air_density0 * exp(-altitude / (R * temperature_K / (M * g)))
	return d


def update_collisions(matrix: hg.Mat4, collisions_object, collisions_raycasts):
	rays_hits = []

	for collision_ray in collisions_raycasts:
		ray_hits = {"name": collision_ray["name"], "hits": []}
		c_pos = matrix * collision_ray["position"]
		c_dir = matrix * (collision_ray["position"] + collision_ray["direction"])
		rc_len = hg.Len(collision_ray["direction"])

		hit = scene_physics.RaycastFirstHit(scene, c_pos, c_dir)

		if 0 < hit.t < rc_len:
			if not collisions_object.test_collision(hit.node):
				ray_hits["hits"].append(hit)
		rays_hits.append(ray_hits)

	terrain_alt, terrain_nrm = get_terrain_altitude(hg.GetT(matrix))

	return rays_hits, terrain_alt, terrain_nrm


def update_physics(matrix, collisions_object, physics_parameters, dts):

	aX = hg.GetX(matrix)
	aY = hg.GetY(matrix)
	aZ = hg.GetZ(matrix)

	# Cap, Pitch & Roll attitude:

	if aY.y > 0:
		y_dir = 1
	else:
		y_dir = -1

	horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
	horizontal_aX = hg.Cross(hg.Vec3.Up, horizontal_aZ) * y_dir
	horizontal_aY = hg.Cross(aZ, horizontal_aX)  # ! It's not an orthogonal repere !

	pitch_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aZ, aZ)))))
	if aZ.y < 0: pitch_attitude *= -1

	roll_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aX, aX)))))
	if aX.y < 0: roll_attitude *= -1

	heading = heading = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aZ, hg.Vec3.Front)))))
	if horizontal_aZ.x < 0:
		heading = 360 - heading

	# axis speed:
	spdX = aX * hg.Dot(aX, physics_parameters["v_move"])
	spdY = aY * hg.Dot(aY, physics_parameters["v_move"])
	spdZ = aZ * hg.Dot(aZ, physics_parameters["v_move"])

	frontal_speed = hg.Len(spdZ)

	# Thrust force:
	k = pow(physics_parameters["thrust_level"], 2) * physics_parameters["thrust_force"]
	# if self.post_combustion and self.thrust_level == 1:
	#    k += self.post_combustion_force
	F_thrust = aZ * k

	pos = hg.GetT(matrix)

	# Air density:
	air_density = compute_atmosphere_density(pos.y)
	# Dynamic pressure:
	q = hg.Vec3(pow(hg.Len(spdX), 2), pow(hg.Len(spdY), 2), pow(hg.Len(spdZ), 2)) * 0.5 * air_density

	# F Lift
	F_lift = aY * q.z * physics_parameters["lift_force"]

	# Drag force:
	F_drag = hg.Normalize(spdX) * q.x * physics_parameters["drag_coefficients"].x + hg.Normalize(spdY) * q.y * physics_parameters["drag_coefficients"].y + hg.Normalize(spdZ) * q.z * physics_parameters["drag_coefficients"].z

	# Total

	physics_parameters["v_move"] += ((F_thrust + F_lift - F_drag) * physics_parameters["health_wreck_factor"] + F_gravity) * dts

	# Displacement:

	pos += physics_parameters["v_move"] * dts

	# Rotations:
	F_pitch = physics_parameters["angular_levels"].x * q.z * physics_parameters["angular_frictions"].x
	F_yaw = physics_parameters["angular_levels"].y * q.z * physics_parameters["angular_frictions"].y
	F_roll = physics_parameters["angular_levels"].z * q.z * physics_parameters["angular_frictions"].z

	# Angular damping:
	gaussian = exp(-pow(frontal_speed * 3.6 * 3 / physics_parameters["speed_ceiling"], 2) / 2)

	# Angular speed:
	angular_speed = hg.Vec3(F_pitch, F_yaw, F_roll) * gaussian

	# Moment:
	pitch_m = aX * angular_speed.x
	yaw_m = aY * angular_speed.y
	roll_m = aZ * angular_speed.z

	# Easy steering:
	if physics_parameters["flag_easy_steering"]:

		easy_yaw_angle = (1 - (hg.Dot(aX, horizontal_aX)))
		if hg.Dot(aZ, hg.Cross(aX, horizontal_aX)) < 0:
			easy_turn_m_yaw = horizontal_aY * -easy_yaw_angle
		else:
			easy_turn_m_yaw = horizontal_aY * easy_yaw_angle

		easy_roll_stab = hg.Cross(aY, horizontal_aY) * y_dir
		if y_dir < 0:
			easy_roll_stab = hg.Normalize(easy_roll_stab)
		else:
			n = hg.Len(easy_roll_stab)
			if n > 0.1:
				easy_roll_stab = hg.Normalize(easy_roll_stab)
				easy_roll_stab *= (1 - n) * n + n * pow(n, 0.125)

		zl = min(1, abs(physics_parameters["angular_levels"].z + physics_parameters["angular_levels"].x + physics_parameters["angular_levels"].y))
		roll_m += (easy_roll_stab * (1 - zl) + easy_turn_m_yaw) * q.z * physics_parameters["angular_frictions"].y * gaussian

	# Moment:
	torque = yaw_m + roll_m + pitch_m
	axis_rot = hg.Normalize(torque)
	moment_speed = hg.Len(torque) * physics_parameters["health_wreck_factor"]

	# Return matrix:

	rot_mat = MathsSupp.rotate_matrix(matrix, axis_rot, moment_speed * dts)
	mat = hg.TransformationMat4(pos, rot_mat)



	return mat, {"v_move": physics_parameters["v_move"], "pitch_attitude": pitch_attitude, "heading": heading, "roll_attitude": roll_attitude}



class Particle:
	def __init__(self, node: hg.Node):
		self.node = node
		node.GetTransform().SetPos(hg.Vec3(0, -1000, 0))
		self.age = -1
		self.v_move = hg.Vec3(0, 0, 0)
		self.delay = 0
		self.scale = 1
		self.rot_speed = hg.Vec3(0, 0, 0)

	def destroy(self, scene):
		scene.DestroyNode(self.node)
		scene.GarbageCollect()

	def kill(self):
		self.age = -1
		self.node.GetTransform().SetPos(hg.Vec3(0, -1000, 0))
		self.node.GetTransform().SetScale(hg.Vec3(0.01, 0.01, 0.01))
		self.node.Disable()

	def get_enabled(self):
		if self.age > 0:
			return True
		else:
			return False


class ParticlesEngine:
	particle_id = 0
	_instances = []
	current_item = 0

	@classmethod
	def reset_engines(cls):
		cls._instances = []

	@classmethod
	def gui(cls):
		generators_list = hg.StringList()
		n = 0
		for engine in cls._instances:
			generators_list.push_back(engine.name)
			n += engine.num_particles

		if hg.ImGuiBegin("Particles settings"):
			hg.ImGuiSetWindowPos("Particles settings", hg.Vec2(680, 60), hg.ImGuiCond_Once)
			hg.ImGuiSetWindowSize("Particles settings", hg.Vec2(450, 930), hg.ImGuiCond_Once)

			hg.ImGuiText("Num generators: %d" % len(cls._instances))
			hg.ImGuiText("Num particles: %d" % n)
			f, d = hg.ImGuiListBox("Generators", cls.current_item, generators_list, 50)
			if f:
				cls.current_item = d
		hg.ImGuiEnd()

	def __init__(self, name, scene, original_node_name, num_particles, start_scale, end_scale, stream_angle, life_time=0., color_label="uColor", write_z=False):
		self.scene = scene
		self.name = name
		ParticlesEngine._instances.append(self)
		self.instance_id = len(ParticlesEngine._instances) - 1
		self.life_time = life_time  # flow life time.
		self.life_t = 0  # Life counter
		self.life_f = 1  # life factor
		self.flow_decrease_date = 0.75  # particles size & alpha decreases life time position (0,1)
		self.color_label = color_label
		self.particles_cnt = 0
		self.particles_cnt_max = 0
		self.particles_cnt_f = 0
		self.num_particles = num_particles
		self.num_alive = 0
		self.flow = 8
		self.particles_delay = 3
		self.particles = []
		self.create_particles(scene.GetNode(original_node_name), write_z)
		self.start_speed_range = hg.Vec2(800, 1200)
		self.delay_range = hg.Vec2(1, 2)
		self.start_scale = start_scale
		self.end_scale = end_scale
		self.scale_range = hg.Vec2(1, 2)
		self.stream_angle = stream_angle
		self.colors = [hg.Color(1, 1, 1, 1), hg.Color(1, 1, 1, 0)]
		self.start_offset = 0
		self.rot_range_x = hg.Vec2(0, 0)
		self.rot_range_y = hg.Vec2(0, 0)
		self.rot_range_z = hg.Vec2(0, 0)
		self.gravity = hg.Vec3(0, -9.8, 0)
		self.linear_damping = 1
		self.loop = True
		self.end = False  # True when loop=True and all particles are dead
		self.num_new = 0
		self.reset()

	def destroy(self):
		for part in self.particles:
			part.destroy(self.scene)
		self.particles = []

	# scene.GarbageCollect()

	def set_rot_range(self, xmin, xmax, ymin, ymax, zmin, zmax):
		self.rot_range_x = hg.Vec2(xmin, xmax)
		self.rot_range_y = hg.Vec2(ymin, ymax)
		self.rot_range_z = hg.Vec2(zmin, zmax)

	def create_particles(self, original_node, write_z):
		for i in range(self.num_particles):
			node = duplicate_node_object(self.scene, original_node, self.name + "." + str(i))
			particle = Particle(node)
			material = particle.node.GetObject().GetMaterial(0)
			hg.SetMaterialWriteZ(material, write_z)
			self.particles.append(particle)

	def deactivate(self):
		for p in self.particles:
			p.node.Disable()

	def reset(self):
		self.num_new = 0
		self.particles_cnt = 0
		self.particles_cnt_f = 0
		self.end = False
		for i in range(self.num_particles):
			self.particles[i].age = -1
			self.particles[i].node.Disable()
			self.particles[i].v_move = hg.Vec3(0, 0, 0)

	def get_direction(self, main_dir):
		if self.stream_angle == 0: return main_dir
		axe0 = hg.Vec3(0, 0, 0)
		axeRot = hg.Vec3(0, 0, 0)
		while hg.Len(axeRot) < 1e-4:
			while hg.Len(axe0) < 1e-5:
				axe0 = hg.Vec3(uniform(-1, 1), uniform(-1, 1), uniform(-1, 1))
			axe0 = hg.Normalize(axe0)
			axeRot = hg.Cross(axe0, main_dir)
		axeRot = hg.Normalize(axeRot)
		return MathsSupp.rotate_vector(main_dir, axeRot, random() * radians(self.stream_angle))

	def update_color(self, particle: Particle):
		if len(self.colors) == 1:
			c = self.colors[0]
		else:
			c = MathsSupp.get_mix_color_value(particle.age / particle.delay, self.colors)
		material = particle.node.GetObject().GetMaterial(0)
		hg.SetMaterialValue(material, self.color_label, hg.Vec4(c.r, c.g, c.b, c.a * self.life_f))

	def reset_life_time(self, life_time=0.):
		self.life_time = life_time
		self.life_t = 0

	def update_kinetics(self, position: hg.Vec3, direction: hg.Vec3, v0: hg.Vec3, axisY: hg.Vec3, dts):

		if self.life_time > 0:
			self.life_t = min(self.life_time, self.life_t + dts)
			if self.life_t >= self.life_time - 1e-6:
				self.end = True
			t = self.life_t / self.life_time
			if t > self.flow_decrease_date:
				self.life_f = 1 - (t - self.flow_decrease_date) / (1 - self.flow_decrease_date)
		else:
			self.life_f = 1

		self.num_new = 0
		if not self.end:
			self.particles_cnt_f += dts * self.flow
			self.num_new = int(self.particles_cnt_f) - self.particles_cnt
			if self.particles_cnt_max > 0:
				if self.num_new + self.particles_cnt > self.particles_cnt_max:
					self.num_new = self.particles_cnt_max - self.particles_cnt
			if self.num_new > 0:
				for i in range(self.num_new):
					if not self.loop and self.particles_cnt + i >= self.num_particles: break
					particle = self.particles[(self.particles_cnt + i) % self.num_particles]
					particle.age = 0
					particle.delay = uniform(self.delay_range.x, self.delay_range.y)
					particle.scale = uniform(self.scale_range.x, self.scale_range.y) * self.life_f
					mat = particle.node.GetTransform()
					dir = self.get_direction(direction)
					rot_mat = hg.Mat3(hg.Cross(axisY, dir), axisY, dir)
					mat.SetPos(position + dir * self.start_offset)
					mat.SetRot(hg.ToEuler(rot_mat))
					mat.SetScale(self.start_scale)
					particle.rot_speed = hg.Vec3(uniform(self.rot_range_x.x, self.rot_range_x.y),
												 uniform(self.rot_range_y.x, self.rot_range_y.y),
												 uniform(self.rot_range_z.x, self.rot_range_z.y))
					particle.v_move = v0 + dir * uniform(self.start_speed_range.x, self.start_speed_range.y)
					particle.node.Disable()
				self.particles_cnt += self.num_new

			n = 0

			for particle in self.particles:
				if particle.age > particle.delay:
					particle.kill()
				elif particle.age == 0:
					particle.age += dts
					n += 1
				elif particle.age > 0:
					n += 1
					if not particle.node.IsEnabled(): particle.node.Enable()
					t = particle.age / particle.delay
					mat = particle.node.GetTransform()
					pos = mat.GetPos()
					rot = mat.GetRot()
					particle.v_move += self.gravity * dts
					spd = hg.Len(particle.v_move)
					particle.v_move -= hg.Normalize(particle.v_move) * spd * self.linear_damping * dts
					pos += particle.v_move * dts
					rot += particle.rot_speed * dts
					pos.y = max(0, pos.y)
					mat.SetPos(pos)
					mat.SetRot(rot)
					mat.SetScale((self.start_scale * (1 - t) + self.end_scale * t) * particle.scale)
					# material = particle.node.GetObject().GetGeometry().GetMaterial(0)
					# material.SetFloat4("self_color",1.,1.,0.,1-t)
					self.update_color(particle)
					# particle.node.GetObject().GetGeometry().GetMaterial(0).SetFloat4("teint", 1,1,1,1)
					particle.age += dts

			self.num_alive = n
			if n == 0 and not self.loop: self.end = True
class LandingTarget:
    def __init__(self, landing_node: hg.Node, horizontal_amplitude=6000, vertical_amplitude=500, smooth_level=5):
        self.landing_node = landing_node
        self.horizontal_amplitude = horizontal_amplitude
        self.vertical_amplitude = vertical_amplitude
        self.smooth_level = smooth_level
        self.extremum = self.calculate_extremum()

    def calculate_extremum(self):
        x = pi / 2
        for i in range(self.smooth_level):
            x = sin(x)
        return x * 2

    def get_position(self, distance):
        org = self.landing_node.GetTransform().GetWorld()
        o = hg.GetT(org)
        az = hg.GetZ(org) * -1
        ah = hg.Normalize(hg.Vec2(az.x, az.z)) * distance
        p = hg.Vec3(ah.x, 0, ah.y) + o
        x = distance / self.horizontal_amplitude
        if x >= 1:
            p.y += self.vertical_amplitude
        elif x > 0:
            x = x * pi - pi / 2
            for i in range(self.smooth_level):
                x = sin(x)
            p.y += (x / self.extremum + 0.5) * self.vertical_amplitude
        return p

    def get_landing_position(self):
        return hg.GetT(self.landing_node.GetTransform().GetWorld())

    def get_approach_entry_position(self):
        return self.get_position(self.horizontal_amplitude)

    def get_landing_vector(self):
        org = self.landing_node.GetTransform().GetWorld()
        az = hg.GetZ(org) * -1
        return hg.Normalize(hg.Vec2(az.x, az.z))

# ==============================================
#       MachineDevice
# ==============================================

class MachineDevice:

    # Start state: activated or not.
    def __init__(self, name, machine, start_state=False):
        self.activated = start_state
        self.start_state = start_state
        self.machine = machine
        self.name = name
        self.wreck = False
        self.commands = {"RESET": self.reset, "ACTIVATE": self.activate, "DEACTIVATE": self.deactivate}

    def record_start_state(self, start_state=None):
        if start_state is None:
            self.start_state = self.activated
        else:
            self.start_state = start_state

    def reset(self):
        self.activated = self.start_state

    def update(self, dts):
        pass

    def activate(self):
        self.activated = True

    def deactivate(self):
        self.activated = False

    def is_activated(self):
        return self.activated


# ==============================================
#       Gear device
# ==============================================

class Gear(MachineDevice):

    def __init__(self, name, machine, scene=None, open_anim=None, retract_anim=None, start_state=True):
        MachineDevice.__init__(self, name, machine, start_state)
        self.flag_gear_moving = False
        self.gear_moving_delay = 3.55
        self.gear_moving_t = 0
        self.gear_level = 1
        self.gear_direction = 0
        self.gear_height = machine.gear_height

        self.scene = scene
        self.open_anim = open_anim
        self.retract_anim = retract_anim
        self.gear_anim_play = None

    def reset(self):
        MachineDevice.reset(self)
        self.flag_gear_moving = False
        if self.activated:
            self.gear_level = 1
        else:
            self.gear_level = 0

        # Animated gear:
        if self.scene is not None:
            if self.gear_anim_play is not None:
                self.scene.StopAnim(self.gear_anim_play)
                self.gear_anim_play = None
            if self.activated:
                # self.gear_moving_delay = Get anim time
                self.gear_anim_play = self.scene.PlayAnim(self.open_anim, hg.ALM_Once, hg.E_Linear, hg.time_from_sec_f(self.gear_moving_delay), hg.time_from_sec_f(self.gear_moving_delay), True, 1)
            else:
                # self.gear_moving_delay = Get anim time
                self.gear_anim_play = self.scene.PlayAnim(self.retract_anim, hg.ALM_Once, hg.E_Linear, hg.time_from_sec_f(self.gear_moving_delay), hg.time_from_sec_f(self.gear_moving_delay), True, 1)

    def activate(self):
        if not self.flag_gear_moving:
            self.start_moving_gear(1)
            self.activated = True

            if self.scene is not None:
                if self.gear_anim_play is not None: self.scene.StopAnim(self.gear_anim_play)
                # self.gear_moving_delay = Get anim time
                self.gear_anim_play = self.scene.PlayAnim(self.open_anim, hg.ALM_Once, hg.E_InOutSine, hg.time_from_sec_f(0), hg.time_from_sec_f(self.gear_moving_delay), False, 1)
                # self.gear_anim_play = self.scene.PlayAnim(self.parent_node.GetInstanceSceneAnim("gear_open"), hg.ALM_Once, hg.E_InOutSine, hg.time_from_sec_f(0), hg.time_from_sec_f(3.55), False, 1)

    def deactivate(self):
        if not self.flag_gear_moving:
            self.start_moving_gear(-1)
            self.activated = False

            if self.scene is not None:
                if self.gear_anim_play is not None: self.scene.StopAnim(self.gear_anim_play)
                # self.gear_moving_delay = Get anim time
                self.gear_anim_play = self.scene.PlayAnim(self.retract_anim, hg.ALM_Once, hg.E_InOutSine, hg.time_from_sec_f(0), hg.time_from_sec_f(self.gear_moving_delay), False, 1)

    def update(self, dts):
        if self.flag_gear_moving:
            lvl = self.gear_moving_t / self.gear_moving_delay
            if self.gear_direction < 0:
                lvl = 1 - lvl
            self.gear_level = max(0, min(1, lvl))
            if self.gear_moving_t < self.gear_moving_delay:
                self.gear_moving_t += dts
            else:
                self.flag_gear_moving = False

    def start_moving_gear(self, direction):  # Deploy: dir = 1, Retract: dir = -1
        self.gear_moving_t = 0
        self.gear_direction = direction
        if self.gear_direction < 0:
            self.gear_level = 1
        else:
            self.gear_level = 0
        self.flag_gear_moving = True

# ==============================================
#       Targetting device
#       Targets and hunter machine are Destroyable_Machine classes only
# ==============================================

class TargettingDevice(MachineDevice):

    def __init__(self, name, machine, start_state=True):
        MachineDevice.__init__(self, name, machine, start_state)
        self.targets = []
        self.target_id = 0
        self.target_lock_range = hg.Vec2(100, 3000)  # Target lock distance range
        self.target_lock_delay = hg.Vec2(1, 5)  # Target lock delay in lock range
        self.target_lock_t = 0
        self.target_locking_state = 0  # 0 to 1
        self.target_locked = False
        self.target_out_of_range = False
        self.target_distance = 0
        self.target_heading = 0
        self.target_altitude = 0
        self.target_angle = 0
        self.destroyable_targets = []
        self.flag_front_lock_cone = True
        self.front_lock_angle = 15

    def set_target_lock_range(self, dmin, dmax):
        self.target_lock_range.x, self.target_lock_range.y = dmin, dmax

    def reset(self):
        self.target_id = 0
        self.target_lock_t = 0
        self.target_locked = False
        self.target_out_of_range = False
        self.target_locking_state = 0

    def get_targets(self):
        return self.targets

    def get_target(self):
        if self.target_id > 0:
            return self.targets[self.target_id - 1]
        else:
            return None

    def get_target_name(self):
        if self.target_id <= 0 or len(self.targets) == 0:
            return ""
        else:
            return self.targets[self.target_id - 1].name

    def get_target_id(self):
        return self.target_id

    def set_target_id(self, tid):
        self.target_id = tid
        if tid > 0:
            if self.targets is None or len(self.targets) == 0:
                self.target_id = 0
            target = self.targets[tid - 1]
            if target.wreck or not target.activated:
                self.next_target()

    def set_target_by_name(self, target_name):
        tid = 0
        for i, tgt in enumerate(self.targets):
            if tgt.name == target_name:
                tid = i + 1
                break
        self.set_target_id(tid)

    def set_destroyable_targets(self, targets):
        self.destroyable_targets = targets

    def search_target(self):
        if len(self.targets) == 0:
            self.target_id == 0
        else:
            self.target_id = int(uniform(0, len(self.targets) - 0.1)) + 1
            target = self.targets[self.target_id - 1]
            if target.wreck or not target.activated:
                self.next_target(False)

    def next_target(self, flag_empty=True):
        if self.targets is not None and len(self.targets) > 0:
            self.target_locked = False
            self.target_lock_t = 0
            self.target_locking_state = 0
            self.target_id += 1
            if self.target_id > len(self.targets):
                if flag_empty:
                    self.target_id = 0
                    return
                else:
                    self.target_id = 1
            t = self.target_id
            target = self.targets[t - 1]
            if target.wreck or not target.activated:
                while target.wreck or not target.activated:
                    self.target_id += 1
                    if self.target_id > len(self.targets):
                        self.target_id = 1
                    if self.target_id == t:
                        self.target_id = 0
                        break
                    target = self.targets[self.target_id - 1]
        else:
            self.target_id = 0

    def update_target_lock(self, dts):
        if self.target_id > 0:
            target = self.targets[self.target_id - 1]
            if target.wreck or not target.activated:
                self.next_target()
                if self.target_id == 0:
                    return
            t_mat, t_pos, t_rot, t_aX, t_aY, t_aZ = self.targets[self.target_id - 1].decompose_matrix()
            mat, pos, rot, aX, aY, dir = self.machine.decompose_matrix()

            v = t_pos - hg.GetT(mat)
            self.target_heading = self.machine.calculate_heading(hg.Normalize(v * hg.Vec3(1, 0, 1)))
            self.target_altitude = t_pos.y
            self.target_distance = hg.Len(v)

            if self.flag_front_lock_cone:
                t_dir = hg.Normalize(v)
                self.target_angle = degrees(acos(max(-1, min(1, hg.Dot(dir, t_dir)))))
                front_lock_angle = self.front_lock_angle
            else:
                self.target_angle = 0
                front_lock_angle = 180

            if self.target_angle < front_lock_angle and self.target_lock_range.x < self.target_distance < self.target_lock_range.y:
                t = (self.target_distance - self.target_lock_range.x) / (
                        self.target_lock_range.y - self.target_lock_range.x)
                delay = self.target_lock_delay.x + t * (self.target_lock_delay.y - self.target_lock_delay.x)
                self.target_out_of_range = False
                self.target_lock_t += dts
                self.target_locking_state = min(1, self.target_lock_t / delay)
                if self.target_lock_t >= delay:
                    self.target_locked = True
            else:
                self.target_locked = False
                self.target_lock_t = 0
                self.target_out_of_range = True
                self.target_locking_state = 0

    def update(self, dts):
        self.update_target_lock(dts)

# =====================================================================================================
#                                   Missiles
# =====================================================================================================

class MissilesDevice(MachineDevice):

    def __init__(self, name, machine, slots_nodes):
        MachineDevice.__init__(self, name, machine, True)
        self.slots_nodes = slots_nodes
        self.num_slots = len(slots_nodes)
        self.missiles_config = []
        self.missiles = [None] * self.num_slots
        self.missiles_started = [None] * self.num_slots
        self.flag_hide_fitted_missiles = False

    def destroy(self):
        if self.missiles is not None:
            for missile in self.missiles:
                if missile is not None:
                    missile.destroy()
        self.missiles = None
        self.num_slots = 0
        self.slots_nodes = None

    def set_missiles_config(self, missiles_config):
        self.missiles_config = missiles_config


    def fit_missile(self, missile, slot_id):
        nd = missile.get_parent_node()
        nd.GetTransform().SetParent(self.slots_nodes[slot_id])
        # print("Fit Missile"+str(slot_id)+" "+str(pos.x)+" "+str(pos.y)+" "+str(pos.z))
        missile.reset(hg.Vec3(0, 0, 0), hg.Vec3(0, 0, 0))
        self.missiles[slot_id] = missile
        if self.flag_hide_fitted_missiles:
            missile.disable_nodes()

    def get_missiles_state(self):
        state = [False] * self.num_slots
        for i in range(self.num_slots):
            if self.missiles[i] is not None:
                state[i] = True
        return state

    def fire_missile(self, slot_id=-1):
        flag_missile_found = False
        missile = None
        if not self.wreck and not self.machine.wreck and slot_id < self.num_slots:
            if slot_id == -1:
                for slot_id in range(self.num_slots):
                    missile = self.missiles[slot_id]
                    if missile is not None and missile.is_armed():
                        flag_missile_found = True
                        break
                if not flag_missile_found:
                    return False, None
            missile = self.missiles[slot_id]
            if missile is not None and missile.is_armed():
                flag_missile_found = True
                self.missiles[slot_id] = None
                trans = missile.get_parent_node().GetTransform()
                mat = trans.GetWorld()
                trans.ClearParent()
                trans.SetWorld(mat)
                td = self.machine.get_device("TargettingDevice")
                if td is not None and td.target_locked:
                    target = td.targets[td.target_id - 1]
                else:
                    target = None
                missile.start(target, self.machine.v_move)
                self.missiles_started[slot_id] = missile

            if self.flag_hide_fitted_missiles:
                if flag_missile_found:
                    missile.enable_nodes()

        return flag_missile_found, missile

    def rearm(self):
        for i in range(self.num_slots):
            if self.missiles[i] is None:
                missile = self.missiles_started[i]
                if missile is not None:
                    missile.deactivate()
                    self.missiles_started[i] = None
                    self.fit_missile(missile, i)

# =====================================================================================================
#                                   Machine Gun
# =====================================================================================================

class MachineGun(MachineDevice):

    def __init__(self, name, machine, slot_node, scene, scene_physics, num_bullets, bullet_node_name="gun_bullet"):
        MachineDevice.__init__(self, name, machine, True)
        self.scene = scene
        self.scene_physics = scene_physics
        self.slot_node = slot_node
        self.bullets_particles = ParticlesEngine(name, scene, bullet_node_name, 24, hg.Vec3(2, 2, 20), hg.Vec3(20, 20, 100), 0.1, 0, "uColor0", True)

        # ParticlesEngine.__init__(self, name, scene, bullet_node_name, 24, hg.Vec3(2, 2, 20), hg.Vec3(20, 20, 100), 0.1, 0, "uColor0", True)

        self.scene_physics = scene_physics
        self.bullets_particles.start_speed_range = hg.Vec2(2000, 2000)
        self.bullets_particles.delay_range = hg.Vec2(2, 2)
        self.bullets_particles.start_offset = 0  # self.start_scale.z
        self.bullets_particles.linear_damping = 0
        self.bullets_particles.scale_range = hg.Vec2(1, 1)
        self.bullets_particles.particles_cnt_max = num_bullets

        self.bullets_feed_backs = []
        #if Destroyable_Machine.flag_activate_particles:
        self.setup_particles()

    def reset(self):
        self.bullets_particles.reset()
        self.bullets_particles.flow = 0

    def setup_particles(self):
        if len(self.bullets_feed_backs) > 0:
            self.destroy_feedbacks()

        for i in range(self.bullets_particles.num_particles):
            fb = ParticlesEngine(self.name + ".fb." + str(i), self.scene, "bullet_impact", 5,
                                 hg.Vec3(1, 1, 1), hg.Vec3(10, 10, 10), 180)
            fb.delay_range = hg.Vec2(1, 1)
            fb.flow = 0
            fb.scale_range = hg.Vec2(1, 3)
            fb.start_speed_range = hg.Vec2(0, 20)
            fb.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., .5, 0.25, 0.25), hg.Color(0.1, 0., 0., 0.)]
            fb.set_rot_range(radians(20), radians(50), radians(10), radians(45), radians(5), radians(15))
            fb.gravity = hg.Vec3(0, 0, 0)
            fb.loop = False
            self.bullets_feed_backs.append(fb)

    def destroy_feedbacks(self):
        for bfb in self.bullets_feed_backs:
            bfb.destroy()
        self.bullets_feed_backs = []

    def destroy_particles(self):
        self.destroy_feedbacks()
        #self.bullets_particles.destroy()

    def destroy_gun(self):
        self.destroy_feedbacks()
        self.bullets_particles.destroy()

    # scene.GarbageCollect()

    def strike(self, i):
        self.bullets_particles.particles[i].kill()
        if len(self.bullets_feed_backs) > 0:
            fb = self.bullets_feed_backs[i]
            fb.reset()
            fb.flow = 3000

    def update(self, dts):
        td = self.machine.get_device("TargettingDevice")
        if td is not None:
            targets = td.destroyable_targets
            mat, pos, rot, aX, axisY, direction = self.machine.decompose_matrix()
            pos_prec = hg.GetT(self.slot_node.GetTransform().GetWorld())
            v0 = self.machine.v_move
            position = pos_prec + v0 * dts

            self.bullets_particles.update_kinetics(position, direction, v0, axisY, dts)
            for i in range(self.bullets_particles.num_particles):
                bullet = self.bullets_particles.particles[i]
                mat = bullet.node.GetTransform()
                pos_fb = mat.GetPos()
                # pos = hg.GetT(mat) - hg.GetZ(mat)

                if bullet.get_enabled():
                    # spd = hg.Len(bullet.v_move)
                    if pos_fb.y < 1:
                        bullet.v_move *= 0
                        self.strike(i)

                    p1 = pos_fb + bullet.v_move

                    #Collision using distance:
                    """
                    for target in targets:
                        distance = hg.Len(target.get_parent_node().GetTransform().GetPos()-pos_fb)
                        if distance < 20: #2 * hg.Len(bullet.v_move) * dts:
                            target.hit(0.1)
                            bullet.v_move = target.v_move
                            self.strike(i)
                            break

                    """
                    rc_len = hg.Len(p1 - pos_fb)
                    hit = self.scene_physics.RaycastFirstHit(self.scene, pos_fb, p1)
                    if 0 < hit.t < rc_len:
                        v_impact = hit.P - pos_fb
                        if hg.Len(v_impact) < 2 * hg.Len(bullet.v_move) * dts:
                            for target in targets:
                                cnds = target.get_collision_nodes()
                                for nd in cnds:
                                    if nd == hit.node:
                                        target.hit(0.1)
                                        bullet.v_move = target.v_move
                                        self.strike(i)
                                        break


                if len(self.bullets_feed_backs) > 0:
                    fb = self.bullets_feed_backs[i]
                    if not fb.end and fb.flow > 0:
                        fb.update_kinetics(pos_fb, hg.Vec3.Front, bullet.v_move, hg.Vec3.Up, dts)

    def get_num_bullets(self):
        return self.bullets_particles.particles_cnt_max - self.bullets_particles.particles_cnt

    def set_num_bullets(self, num):
        self.bullets_particles.particles_cnt_max = int(num)
        self.bullets_particles.reset()

    def fire_machine_gun(self):
        if not self.wreck:
            self.bullets_particles.flow = 24 / 2

    def stop_machine_gun(self):
        self.bullets_particles.flow = 0

    def is_gun_activated(self):
        if self.bullets_particles.flow == 0:
            return False
        else:
            return True

    def get_new_bullets_count(self):
        return self.bullets_particles.num_new

# ==============================================
#       Control device
# ==============================================

class ControlDevice(MachineDevice):

    CM_KEYBOARD = "Keyboard"
    CM_GAMEPAD = "GamePad"
    CM_MOUSE = "Mouse"
    CM_LOGITECH_EXTREME_3DPRO = "Logitech extreme 3DPro"
    CM_LOGITECH_ATTACK_3 = "Logitech Attack 3"

    keyboard = None
    mouse = None
    gamepad = None
    generic_controller = None

    @classmethod
    def init(cls, keyboard, mouse, gamepad, generic_controller):
        cls.keyboard = keyboard
        cls.mouse = mouse
        cls.gamepad = gamepad
        cls.generic_controller = generic_controller

    def __init__(self, name, machine, inputs_mapping_file="", input_mapping_name="", control_mode=CM_KEYBOARD, start_state=False):
        MachineDevice.__init__(self, name, machine, start_state)
        self.flag_user_control = True
        self.control_mode = control_mode
        self.inputs_mapping_file = inputs_mapping_file
        self.inputs_mapping_encoded = {}
        self.inputs_mapping = {}
        self.input_mapping_name = input_mapping_name
        if self.inputs_mapping_file != "":
            self.load_inputs_mapping_file(self.inputs_mapping_file)

    def set_control_mode(self, cmode):
        self.control_mode = cmode

    def load_inputs_mapping_file(self, file_name):
        file = hg.OpenText(file_name)
        if not file:
            print("ERROR - Can't open json file : " + file_name)
        else:
            json_script = hg.ReadString(file)
            hg.Close(file)
            if json_script != "":
                self.inputs_mapping_encoded = json.loads(json_script)
                im = self.inputs_mapping_encoded[self.input_mapping_name]
                cmode_decode = {}
                for cmode, maps in im.items():
                    maps_decode = {}
                    for cmd, hg_enum in maps.items():
                        if hg_enum != "":
                            if not hg_enum.isdigit():
                                try:
                                    exec("maps_decode['%s'] = hg.%s" % (cmd, hg_enum))
                                except AttributeError:
                                    print("ERROR - Harfang Enum not implemented ! - " + "hg." + hg_enum)
                                    maps_decode[cmd] = ""
                            else:
                                maps_decode[cmd] = int(hg_enum)
                        else:
                            maps_decode[cmd] = ""
                    cmode_decode[cmode] = maps_decode
                self.inputs_mapping = {self.input_mapping_name: cmode_decode}
            else:
                print("ERROR - Inputs parameters empty : " + file_name)

    def activate_user_control(self):
        self.flag_user_control = True

    def deactivate_user_control(self):
        self.flag_user_control = False

    def is_user_control_active(self):
        return self.flag_user_control

# ==============================================
#       Missile user control device - Dubug mode
# ==============================================

class MissileUserControlDevice(ControlDevice):

    def __init__(self, name, machine, control_mode=ControlDevice.CM_KEYBOARD, start_state=False):
        ControlDevice.__init__(self, name, machine, "", "MissileLauncherUserInputsMapping", control_mode, start_state)
        self.pos_mem = None

    def update(self, dts):
        if self.is_activated():
            mat, pos, rot, aX, aY, aZ = self.machine.decompose_matrix()
            step = 0.5
            if ControlDevice.keyboard.Down(hg.K_Up):
                pos += aY * step
            if ControlDevice.keyboard.Down(hg.K_Down):
                pos -= aY * step
            if ControlDevice.keyboard.Down(hg.K_Left):
                pos -= aX * step
            if ControlDevice.keyboard.Down(hg.K_Right):
                pos += aX * step
            if ControlDevice.keyboard.Down(hg.K_Add):
                pos += aZ * step
            if ControlDevice.keyboard.Down(hg.K_Sub):
                pos -= aZ * step
            self.pos_mem = pos


# ==============================================
#       Missile launcher user control device
# ==============================================

class MissileLauncherUserControlDevice(ControlDevice):

    def __init__(self, name, machine, inputs_mapping_file, control_mode=ControlDevice.CM_KEYBOARD, start_state=False):
        ControlDevice.__init__(self, name, machine, inputs_mapping_file, "MissileLauncherUserInputsMapping", control_mode, start_state)
        self.set_control_mode(control_mode)

    def set_control_mode(self, cmode):
        self.control_mode = cmode
        if cmode == ControlDevice.CM_KEYBOARD:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_kb,
                "NEXT_PILOT": self.next_pilot_kb,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_kb,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_kb,
                "NEXT_TARGET": self.next_target_kb,
                "FIRE_MISSILE": self.fire_missile_kb,
                "REARM": self.rearm_kb
            })
        elif cmode == ControlDevice.CM_GAMEPAD:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_gp,
                "NEXT_PILOT": self.next_pilot_gp,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_gp,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_gp,
                "NEXT_TARGET": self.next_target_gp,
                "FIRE_MISSILE": self.fire_missile_gp,
                "REARM": self.rearm_gp
            })
        elif cmode == ControlDevice.CM_LOGITECH_ATTACK_3:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_la3,
                "NEXT_PILOT": self.next_pilot_la3,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_la3,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_la3,
                "NEXT_TARGET": self.next_target_la3,
                "FIRE_MISSILE": self.fire_missile_la3,
                "REARM": self.rearm_la3
            })

        elif cmode == ControlDevice.CM_LOGITECH_EXTREME_3DPRO:
            self.commands.update({})

        elif cmode == ControlDevice.CM_MOUSE:
            self.commands.update({})

    # ====================================================================================

    def update_cm_la3(self, dts):
        im = self.inputs_mapping["MissileLauncherUserInputsMapping"]["LogitechAttack3"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_keyboard(self, dts):
        im = self.inputs_mapping["MissileLauncherUserInputsMapping"]["Keyboard"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_gamepad(self, dts):
        im = self.inputs_mapping["MissileLauncherUserInputsMapping"]["GamePad"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_mouse(self, dts):
        im = self.inputs_mapping["MissileLauncherUserInputsMapping"]["Mouse"]

    def update(self, dts):
        if self.activated:
            if self.flag_user_control and self.machine.has_focus():
                if self.control_mode == ControlDevice.CM_KEYBOARD:
                    self.update_cm_keyboard(dts)
                elif self.control_mode == ControlDevice.CM_GAMEPAD:
                    self.update_cm_gamepad(dts)
                elif self.control_mode == ControlDevice.CM_MOUSE:
                    self.update_cm_mouse(dts)
                elif self.control_mode == ControlDevice.CM_LOGITECH_ATTACK_3:
                    self.update_cm_la3(dts)

    # =============================== Keyboard commands ====================================

    def switch_activation_kb(self, value):
        pass

    def next_pilot_kb(self, value):
        pass

    def increase_health_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_health_level(self.machine.health_level + 0.01)

    def decrease_health_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_health_level(self.machine.health_level - 0.01)

    def next_target_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def fire_missile_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()

    def rearm_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            self.machine.rearm()

    # =============================== Logitech Attack 3 ====================================

    def switch_activation_la3(self, value):
        pass

    def next_pilot_la3(self, value):
        pass

    def increase_health_level_la3(self, value):
        pass

    def decrease_health_level_la3(self, value):
        pass

    def next_target_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def fire_missile_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()

    def rearm_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            self.machine.rearm()

    # =============================== Gamepad commands ====================================

    def switch_activation_gp(self, value):
        pass

    def next_pilot_gp(self, value):
        pass

    def increase_health_level_gp(self, value):
        pass

    def decrease_health_level_gp(self, value):
        pass

    def next_target_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def fire_missile_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()

    def rearm_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            self.machine.rearm()

# ==============================================
#       Aircraft user control device
# ==============================================

class AircraftUserControlDevice(ControlDevice):

    def __init__(self, name, machine, inputs_mapping_file, control_mode=ControlDevice.CM_KEYBOARD, start_state=False):
        ControlDevice.__init__(self, name, machine, inputs_mapping_file, "AircraftUserInputsMapping", control_mode, start_state)
        self.set_control_mode(control_mode)

    def set_control_mode(self, cmode):
        self.control_mode = cmode
        if cmode == ControlDevice.CM_KEYBOARD:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_kb,
                "NEXT_PILOT": self.next_pilot_kb,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_kb,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_kb,
                "INCREASE_THRUST_LEVEL": self.increase_thrust_level_kb,
                "DECREASE_THRUST_LEVEL": self.decrease_thrust_level_kb,
                "SET_THRUST_LEVEL": self.set_thrust_level_kb,
                "INCREASE_BRAKE_LEVEL": self.increase_brake_level_kb,
                "DECREASE_BRAKE_LEVEL": self.decrease_brake_level_kb,
                "INCREASE_FLAPS_LEVEL": self.increase_flaps_level_kb,
                "DECREASE_FLAPS_LEVEL": self.decrease_flaps_level_kb,
                "ROLL_LEFT": self.roll_left_kb,
                "ROLL_RIGHT": self.roll_right_kb,
                "SET_ROLL": self.set_roll_kb,
                "PITCH_UP": self.pitch_up_kb,
                "PITCH_DOWN": self.pitch_down_kb,
                "SET_PITCH": self.set_pitch_kb,
                "YAW_LEFT": self.yaw_left_kb,
                "YAW_RIGHT": self.yaw_right_kb,
                "SET_YAW": self.set_yaw_kb,
                "SWITCH_POST_COMBUSTION": self.switch_post_combustion_kb,
                "NEXT_TARGET": self.next_target_kb,
                "SWITCH_GEAR": self.switch_gear_kb,
                "ACTIVATE_IA": self.activate_ia_kb,
                "ACTIVATE_AUTOPILOT": self.activate_autopilot_kb,
                "SWITCH_EASY_STEERING": self.switch_easy_steering_kb,
                "FIRE_MACHINE_GUN": self.fire_machine_gun_kb,
                "FIRE_MISSILE": self.fire_missile_kb,
                "REARM": self.rearm_kb
            })

        elif cmode == ControlDevice.CM_GAMEPAD:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_gp,
                "NEXT_PILOT": self.next_pilot_gp,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_gp,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_gp,
                "INCREASE_THRUST_LEVEL": self.increase_thrust_level_gp,
                "DECREASE_THRUST_LEVEL": self.decrease_thrust_level_gp,
                "SET_THRUST_LEVEL": self.set_thrust_level_gp,
                "INCREASE_BRAKE_LEVEL": self.increase_brake_level_gp,
                "DECREASE_BRAKE_LEVEL": self.decrease_brake_level_gp,
                "INCREASE_FLAPS_LEVEL": self.increase_flaps_level_gp,
                "DECREASE_FLAPS_LEVEL": self.decrease_flaps_level_gp,
                "ROLL_LEFT": self.roll_left_gp,
                "ROLL_RIGHT": self.roll_right_gp,
                "SET_ROLL": self.set_roll_gp,
                "PITCH_UP": self.pitch_up_gp,
                "PITCH_DOWN": self.pitch_down_gp,
                "SET_PITCH": self.set_pitch_gp,
                "YAW_LEFT": self.yaw_left_gp,
                "YAW_RIGHT": self.yaw_right_gp,
                "SET_YAW": self.set_yaw_gp,
                "SWITCH_POST_COMBUSTION": self.switch_post_combustion_gp,
                "NEXT_TARGET": self.next_target_gp,
                "SWITCH_GEAR": self.switch_gear_gp,
                "ACTIVATE_AUTOPILOT": self.activate_autopilot_gp,
                "ACTIVATE_IA": self.activate_ia_gp,
                "SWITCH_EASY_STEERING": self.switch_easy_steering_gp,
                "FIRE_MACHINE_GUN": self.fire_machine_gun_gp,
                "FIRE_MISSILE": self.fire_missile_gp})

        elif cmode == ControlDevice.CM_LOGITECH_ATTACK_3:
            self.commands.update({
                "SWITCH_ACTIVATION": self.switch_activation_la3,
                "NEXT_PILOT": self.next_pilot_la3,
                "INCREASE_HEALTH_LEVEL": self.increase_health_level_la3,
                "DECREASE_HEALTH_LEVEL": self.decrease_health_level_la3,
                "INCREASE_THRUST_LEVEL": self.increase_thrust_level_la3,
                "DECREASE_THRUST_LEVEL": self.decrease_thrust_level_la3,
                "SET_THRUST_LEVEL": self.set_thrust_level_la3,
                "INCREASE_BRAKE_LEVEL": self.increase_brake_level_la3,
                "DECREASE_BRAKE_LEVEL": self.decrease_brake_level_la3,
                "INCREASE_FLAPS_LEVEL": self.increase_flaps_level_la3,
                "DECREASE_FLAPS_LEVEL": self.decrease_flaps_level_la3,
                "ROLL_LEFT": self.roll_left_la3,
                "ROLL_RIGHT": self.roll_right_la3,
                "SET_ROLL": self.set_roll_la3,
                "PITCH_UP": self.pitch_up_la3,
                "PITCH_DOWN": self.pitch_down_la3,
                "SET_PITCH": self.set_pitch_la3,
                "YAW_LEFT": self.yaw_left_la3,
                "YAW_RIGHT": self.yaw_right_la3,
                "SET_YAW": self.set_yaw_la3,
                "SWITCH_POST_COMBUSTION": self.switch_post_combustion_la3,
                "NEXT_TARGET": self.next_target_la3,
                "SWITCH_GEAR": self.switch_gear_la3,
                "ACTIVATE_AUTOPILOT": self.activate_autopilot_la3,
                "ACTIVATE_IA": self.activate_ia_la3,
                "SWITCH_EASY_STEERING": self.switch_easy_steering_la3,
                "FIRE_MACHINE_GUN": self.fire_machine_gun_la3,
                "FIRE_MISSILE": self.fire_missile_la3
            })

        elif cmode == ControlDevice.CM_LOGITECH_EXTREME_3DPRO:
            self.commands.update({
                })

        elif cmode == ControlDevice.CM_MOUSE:
            self.commands.update({})

    # =================== Functions =================================================================

    def update_cm_la3(self, dts):
        im = self.inputs_mapping["AircraftUserInputsMapping"]["LogitechAttack3"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_keyboard(self, dts):
        im = self.inputs_mapping["AircraftUserInputsMapping"]["Keyboard"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update(self, dts):
        if self.activated:
            if get_connected()[0]:
                self.machine.set_yaw_level(get_thumb_values(get_state(0))[0][0])
                self.machine.set_thrust_level(max(0, min(get_thumb_values(get_state(0))[0][1]*1.00793651, 1)))
                self.machine.set_brake_level(max(0, -1*get_thumb_values(get_state(0))[0][1]))
                self.machine.set_roll_level(-1*get_thumb_values(get_state(0))[1][0])
                self.machine.set_pitch_level(get_thumb_values(get_state(0))[1][1])
                if get_button_values(get_state(0))['Y']:
                    if "Gear" in self.machine.devices and self.machine.devices["Gear"] is not None:
                        gear = self.machine.devices["Gear"]
                        if not self.machine.flag_landed:
                            if gear.activated:
                                gear.deactivate()
                            else:
                                gear.activate()
                if get_button_values(get_state(0))['A']:
                    n = self.machine.get_machinegun_count()
                    for i in range(n):
                        mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                        if mgd is not None and not mgd.is_gun_activated():
                            mgd.fire_machine_gun()
                else:
                    n = self.machine.get_machinegun_count()
                    for i in range(n):
                        mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                        if mgd is not None and mgd.is_gun_activated():
                            mgd.stop_machine_gun()
                if get_button_values(get_state(0))['X']:
                    md = self.machine.get_device("MissilesDevice")
                    if md is not None:
                        md.fire_missile()
                if get_button_values(get_state(0))['B']:
                    ia_device = self.machine.get_device("IAControlDevice")
                    if ia_device is not None:
                        self.deactivate()
                        ia_device.activate()
            elif self.flag_user_control and self.machine.has_focus():
                if self.control_mode == ControlDevice.CM_KEYBOARD:
                    self.update_cm_keyboard(dts)
                elif self.control_mode == ControlDevice.CM_LOGITECH_ATTACK_3:
                    self.update_cm_la3(dts)
            if self.machine.get_thrust_level()>=0.9:
                self.machine.activate_post_combustion()
            else:
                self.machine.deactivate_post_combustion()
    # =============================== Keyboard commands ====================================

    def switch_activation_kb(self, value):
        pass
    def switch_post_combustion_kb(self, value):
        pass
    def next_pilot_kb(self, value):
        pass

    def increase_health_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_health_level(self.machine.health_level + 0.01)

    def decrease_health_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_health_level(self.machine.health_level - 0.01)

    def increase_thrust_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_thrust_level(self.machine.thrust_level_dest + 0.01)

    def decrease_thrust_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_thrust_level(self.machine.thrust_level_dest - 0.01)

    def set_thrust_level_kb(self, value):
        pass

    def increase_brake_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_brake_level(1)
        else:
            self.machine.set_brake_level(0)
    def decrease_brake_level_kb(self, value):
        pass
    def increase_flaps_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level + 0.01)

    def decrease_flaps_level_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level - 0.01)

    def roll_left_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_roll_level(1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_roll_level(0)

    def roll_right_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_roll_level(-1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_roll_level(0)

    def set_roll_kb(self, value):
        pass

    def pitch_up_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_pitch_level(1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_pitch_level(0)

    def pitch_down_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_pitch_level(-1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_pitch_level(0)

    def set_pitch_kb(self, value):
        pass

    def yaw_left_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_yaw_level(-1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_yaw_level(0)

    def yaw_right_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.machine.set_yaw_level(1)
        elif ControlDevice.keyboard.Released(value):
            self.machine.set_yaw_level(0)

    def set_yaw_kb(self, value):
        pass
    def next_target_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def switch_gear_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            if "Gear" in self.machine.devices and self.machine.devices["Gear"] is not None:
                gear = self.machine.devices["Gear"]
                if not self.machine.flag_landed:
                    if gear.activated:
                        gear.deactivate()
                    else:
                        gear.activate()

    def activate_autopilot_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            autopilot_device = self.machine.get_device("AutopilotControlDevice")
            if autopilot_device is not None:
                self.deactivate()
                autopilot_device.activate()


    def activate_ia_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            ia_device = self.machine.get_device("IAControlDevice")
            if ia_device is not None:
                self.deactivate()
                ia_device.activate()

    def switch_easy_steering_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            self.machine.flag_easy_steering = not self.machine.flag_easy_steering

    def fire_machine_gun_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and not mgd.is_gun_activated():
                    mgd.fire_machine_gun()
        elif ControlDevice.keyboard.Released(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and mgd.is_gun_activated():
                    mgd.stop_machine_gun()

    def fire_missile_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()

    def rearm_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            self.machine.rearm()

    # =============================== Logitech Attack 3 ====================================

    def switch_activation_la3(self, value):
        pass

    def next_pilot_la3(self, value):
        pass

    def increase_health_level_la3(self, value):
        pass

    def decrease_health_level_la3(self, value):
        pass

    def increase_thrust_level_la3(self, value):
        pass

    def decrease_thrust_level_la3(self, value):
        pass

    def set_thrust_level_la3(self, value):
        # epsilon = 0.1 # threshold to cancel the device's jitter
        v = -ControlDevice.generic_controller.Axes(value)
        self.machine.set_thrust_level((v + 1.0) / 2.0)

    def increase_brake_level_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_brake_level(self.machine.brake_level_dest + 0.01)

    def decrease_brake_level_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_brake_level(self.machine.brake_level_dest - 0.01)

    def increase_flaps_level_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level_dest + 0.01)

    def decrease_flaps_level_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level_dest - 0.01)

    def roll_left_la3(self, value):
        pass

    def roll_right_la3(self, value):
        pass

    def set_roll_la3(self, value):
        v = -ControlDevice.generic_controller.Axes(value)
        self.machine.set_roll_level(v)

    def set_pitch_la3(self, value):
        v = -ControlDevice.generic_controller.Axes(value)
        self.machine.set_pitch_level(v)

    def pitch_up_la3(self, value):
        pass

    def pitch_down_la3(self, value):
        pass

    def yaw_left_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_yaw_level(-1)
        elif ControlDevice.generic_controller.Released(value):
            self.machine.set_yaw_level(0)

    def yaw_right_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            self.machine.set_yaw_level(1)
        elif ControlDevice.generic_controller.Released(value):
            self.machine.set_yaw_level(0)

    def set_yaw_la3(self, value):
        pass

    def switch_post_combustion_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            if self.machine.post_combustion:
                self.machine.deactivate_post_combustion()
            else:
                self.machine.activate_post_combustion()

    def next_target_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def switch_gear_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            if "Gear" in self.machine.devices and self.machine.devices["Gear"] is not None:
                gear = self.machine.devices["Gear"]
                if not self.machine.flag_landed:
                    if gear.activated:
                        gear.deactivate()
                    else:
                        gear.activate()


    def activate_autopilot_la3(self, value):
        pass

    def activate_ia_la3(self, value):
        pass

    def switch_easy_steering_la3(self, value):
        pass


    def fire_machine_gun_la3(self, value):
        if ControlDevice.generic_controller.Down(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and not mgd.is_gun_activated():
                    mgd.fire_machine_gun()
        elif ControlDevice.generic_controller.Released(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and mgd.is_gun_activated():
                    mgd.stop_machine_gun()


    def fire_missile_la3(self, value):
        if ControlDevice.generic_controller.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()



    # =============================== Gamepad commands ====================================

    def switch_activation_gp(self, value):
        pass

    def next_pilot_gp(self, value):
        pass

    def increase_health_level_gp(self, value):
        pass

    def decrease_health_level_gp(self, value):
        pass

    def increase_thrust_level_gp(self, value):
        pass

    def decrease_thrust_level_gp(self, value):
        pass

    def set_thrust_level_gp(self, value):
        epsilon = 0.1
        v = -ControlDevice.gamepad.Axes(value)
        if v < - epsilon or v > epsilon:
            self.machine.set_thrust_level(self.machine.thrust_level_dest + v * 0.01)

    def increase_brake_level_gp(self, value):
        if ControlDevice.gamepad.Down(value):
            self.machine.set_brake_level(self.machine.brake_level_dest + 0.01)

    def decrease_brake_level_gp(self, value):
        if ControlDevice.gamepad.Down(value):
            self.machine.set_brake_level(self.machine.brake_level_dest - 0.01)

    def increase_flaps_level_gp(self, value):
        if ControlDevice.gamepad.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level_dest + 0.01)

    def decrease_flaps_level_gp(self, value):
        if ControlDevice.gamepad.Down(value):
            self.machine.set_flaps_level(self.machine.flaps_level_dest - 0.01)

    def roll_left_gp(self, value):
        pass

    def roll_right_gp(self, value):
        pass

    def set_roll_gp(self, value):
        v = -ControlDevice.gamepad.Axes(value)
        self.machine.set_roll_level(v)

    def pitch_up_gp(self, value):
        pass

    def pitch_down_gp(self, value):
        pass

    def set_pitch_gp(self, value):
        v = -ControlDevice.gamepad.Axes(value)
        self.machine.set_pitch_level(v)

    def yaw_left_gp(self, value):
        pass

    def yaw_right_gp(self, value):
        pass

    def set_yaw_gp(self, value):
        epsilon = 0.016
        v = ControlDevice.gamepad.Axes(value)
        if -epsilon < v < epsilon:
            v = 0
        self.machine.set_yaw_level(v)

    def switch_post_combustion_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            if self.machine.post_combustion:
                self.machine.deactivate_post_combustion()
            else:
                self.machine.activate_post_combustion()

    def next_target_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            td = self.machine.get_device("TargettingDevice")
            if td is not None:
                td.next_target()

    def switch_gear_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            if "Gear" in self.machine.devices and self.machine.devices["Gear"] is not None:
                gear = self.machine.devices["Gear"]
                if not self.machine.flag_landed:
                    if gear.activated:
                        gear.deactivate()
                    else:
                        gear.activate()

    def activate_autopilot_gp(self, value):
        pass

    def activate_ia_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            ia_device = self.machine.get_device("IAControlDevice")
            if ia_device is not None:
                self.deactivate()
                ia_device.activate()

    def switch_easy_steering_gp(self, value):
        pass

    def fire_machine_gun_gp(self, value):
        if ControlDevice.gamepad.Down(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and not mgd.is_gun_activated():
                    mgd.fire_machine_gun()
        elif ControlDevice.gamepad.Released(value):
            n = self.machine.get_machinegun_count()
            for i in range(n):
                mgd = self.machine.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and mgd.is_gun_activated():
                    mgd.stop_machine_gun()

    def fire_missile_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            md = self.machine.get_device("MissilesDevice")
            if md is not None:
                md.fire_missile()


# ===================================================================
#       Aircraft Autopilot control device
#       Autopilot device con controls only one aircraft
# ===================================================================

class AircraftAutopilotControlDevice(ControlDevice):

    def __init__(self, name, machine, inputs_mapping_file, control_mode=ControlDevice.CM_KEYBOARD, start_state=False):
        ControlDevice.__init__(self, name, machine, inputs_mapping_file, "AircraftAutopilotInputsMapping", control_mode, start_state)

        self.autopilot_speed = -1  # m.s-1
        self.autopilot_heading = 0  # degrees
        self.autopilot_altitude = 500  # m

        self.autopilot_roll_attitude = 0
        self.autopilot_pitch_attitude = 0

        self.heading_step = 1

        self.altitude_step = 10
        self.altitude_range = [0, 50000]

        self.speed_step = 10
        self.speed_range = [-1, 3000 * 3.6]

        self.flag_easy_steering_mem = False

        self.set_control_mode(control_mode)

    def set_control_mode(self, cmode):
        self.control_mode = cmode
        if cmode == ControlDevice.CM_KEYBOARD:
            self.commands.update({
                "ACTIVATE_USER_CONTROL": self.activate_user_control_kb,
                "INCREASE_SPEED": self.increase_speed_kb,
                "DECREASE_SPEED": self.decrease_speed_kb,
                "SET_SPEED": self.set_speed_kb,
                "INCREASE_HEADING": self.increase_heading_kb,
                "DECREASE_HEADING": self.decrease_heading_kb,
                "SET_HEADING": self.set_heading_kb,
                "INCREASE_ALTITUDE": self.increase_altitude_kb,
                "DECREASE_ALTITUDE": self.decrease_altitude_kb,
                "SET_ALTITUDE": self.set_altitude_kb
            })

        elif cmode == ControlDevice.CM_GAMEPAD:
            self.commands.update({
                "ACTIVATE_USER_CONTROL": self.activate_user_control_gp,
                "INCREASE_SPEED": self.increase_speed_gp,
                "DECREASE_SPEED": self.decrease_speed_gp,
                "SET_SPEED": self.set_speed_gp,
                "INCREASE_HEADING": self.increase_heading_gp,
                "DECREASE_HEADING": self.decrease_heading_gp,
                "SET_HEADING": self.set_heading_gp,
                "INCREASE_ALTITUDE": self.increase_altitude_gp,
                "DECREASE_ALTITUDE": self.decrease_altitude_gp,
                "SET_ALTITUDE": self.set_altitude_gp
            })

        elif cmode == ControlDevice.CM_MOUSE:
            self.commands.update({})


    # ============================== functions

    def activate(self):
        if not self.activated:
            ControlDevice.activate(self)
            self.flag_easy_steering_mem = self.machine.flag_easy_steering
            self.machine.flag_easy_steering = True

    def deactivate(self):
        if self.activated:
            ControlDevice.deactivate(self)
            self.machine.flag_easy_steering = self.flag_easy_steering_mem

    def set_autopilot_speed(self, value):
        self.autopilot_speed = value

    def set_autopilot_heading(self, value):
        self.autopilot_heading = max(min(360, value), 0)

    def set_autopilot_altitude(self, value):
        self.autopilot_altitude = value

    # =============================== Keyboard commands ====================================

    def activate_user_control_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            uctrl = self.machine.get_device("UserControlDevice")
            if uctrl is not None:
                self.deactivate()
                uctrl.activate()

    def increase_speed_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_speed = min(self.autopilot_speed + self.speed_step, self.speed_range[1])

    def decrease_speed_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_speed = max(self.autopilot_speed - self.speed_step, self.speed_range[0])

    def set_speed_kb(self, value):
        pass

    def increase_heading_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_heading = (self.autopilot_heading + self.heading_step) % 360

    def decrease_heading_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_heading = (self.autopilot_heading - self.heading_step) % 360

    def set_heading_kb(self, value):
        pass

    def increase_altitude_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_altitude = min(self.autopilot_altitude + self.altitude_step, self.altitude_range[1])

    def decrease_altitude_kb(self, value):
        if ControlDevice.keyboard.Down(value):
            self.autopilot_altitude = max(self.autopilot_altitude - self.altitude_step, self.altitude_range[0])

    def set_altitude_kb(self, value):
        pass

    # =============================== Gamepad commands ====================================

    def switch_activation_gp(self, value):
        pass

    def increase_speed_gp(self, value):
        pass

    def decrease_speed_gp(self, value):
        pass

    def set_speed_kb(self, value):
        pass

    def increase_heading_gp(self, value):
        pass

    def decrease_heading_gp(self, value):
        pass

    def set_heading_gp(self, value):
        pass

    def increase_altitude_gp(self, value):
        pass

    def decrease_altitude_gp(self, value):
        pass

    def set_altitude_gp(self, value):
        pass

    # ====================================================================================

    def update_cm_la3(self, dts):
        im = self.inputs_mapping["AircraftAutopilotInputsMapping"]["LogitechAttack3"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_keyboard(self, dts):
        im = self.inputs_mapping["AircraftAutopilotInputsMapping"]["Keyboard"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_gamepad(self, dts):
        im = self.inputs_mapping["AircraftAutopilotInputsMapping"]["GamePad"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_mouse(self, dts):
        im = self.inputs_mapping["AircraftAutopilotInputsMapping"]["Mouse"]

    def update_controlled_devices(self, dts):
        aircraft = self.machine
        if not aircraft.wreck and not aircraft.flag_going_to_takeoff_position and not aircraft.flag_landed:
            if self.autopilot_speed >= 0:
                a_range = 1
                v = aircraft.get_linear_speed()
                a = aircraft.get_linear_acceleration()
                f = v / self.autopilot_speed * 100

                if f < 80:
                    aircraft.set_brake_level(0)
                    # self.set_flaps_level(0)
                    aircraft.set_thrust_level(1)
                    aircraft.activate_post_combustion()
                elif f > 120:
                    aircraft.set_thrust_level(0.25)
                    aircraft.set_brake_level(0.5)
                elif f < 100:
                    if a < -a_range:
                        aircraft.set_brake_level(0)
                        # self.set_flaps_level(0)
                        aircraft.set_thrust_level(1)
                        aircraft.activate_post_combustion()
                    else:
                        # fa = 1+(a/a_range) #1 - ((f - 80) / 20)
                        fa = 1 - ((f - 80) / 20)
                        aircraft.set_thrust_level(0.5 + fa * 0.5)
                elif f > 100:
                    if a > a_range:
                        aircraft.set_thrust_level(0.25)
                        aircraft.set_brake_level(0.5)
                    else:
                        fa = (120 - f) / 20
                        # fa = max(0,(a/a_range)) #(120-f)/20
                        aircraft.set_brake_level(0.5 * fa)
                        aircraft.set_thrust_level(0.75)
                # vkm=v*3.6
                # if vkm<self.landing_max_speed:
                #    f=1-max(0,min(1,(vkm-self.minimum_flight_speed)/(self.landing_max_speed-self.minimum_flight_speed)))
                #    self.set_flaps_level(self.f)

            # straighten aircraft:
            mat = aircraft.parent_node.GetTransform().GetWorld()
            aY = hg.GetY(mat)
            if aY.y < 0:
                aircraft.set_roll_level(0)
                aircraft.set_pitch_level(0)
                aircraft.set_yaw_level(0)
            else:
                # heading / roll_attitude:
                diff = self.autopilot_heading - aircraft.heading
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                tc = max(-1, min(1, -diff / 90))
                if tc < 0:
                    tc = -pow(-tc, 0.5)
                else:
                    tc = pow(tc, 0.5)

                self.autopilot_roll_attitude = max(min(180, tc * 85), -180)

                diff = self.autopilot_roll_attitude - aircraft.roll_attitude
                tr = max(-1, min(1, diff / 20))
                aircraft.set_roll_level(tr)

                # altitude / pitch_attitude:
                diff = self.autopilot_altitude - aircraft.get_altitude()
                ta = max(-1, min(1, diff / 500))

                if ta < 0:
                    ta = -pow(-ta, 0.7)
                else:
                    ta = pow(ta, 0.7)

                self.autopilot_pitch_attitude = max(min(180, ta * 45), -180)

                diff = self.autopilot_pitch_attitude - aircraft.pitch_attitude
                tp = max(-1, min(1, diff / 10))
                aircraft.set_pitch_level(-tp)

    def update(self, dts):
        if self.activated:
            if self.flag_user_control and self.machine.has_focus():
                if self.control_mode == ControlDevice.CM_KEYBOARD:
                    self.update_cm_keyboard(dts)
                elif self.control_mode == ControlDevice.CM_GAMEPAD:
                    self.update_cm_gamepad(dts)
                elif self.control_mode == ControlDevice.CM_MOUSE:
                    self.update_cm_mouse(dts)
                elif self.control_mode == ControlDevice.CM_LOGITECH_ATTACK_3:
                     self.update_cm_la3(dts)

            self.update_controlled_devices(dts)


# ===========================================================
#       Aircraft IA control device
#       IA control device can control only one aicraft.
# ===========================================================


class AircraftIAControlDevice(ControlDevice):
    IA_COM_IDLE = 0
    IA_COM_LIFTOFF = 1
    IA_COM_FIGHT = 2
    IA_COM_RETURN_TO_BASE = 3
    IA_COM_LANDING = 4

    def __init__(self, name, machine, inputs_mapping_file, control_mode=ControlDevice.CM_KEYBOARD, start_state=False):
        ControlDevice.__init__(self, name, machine, inputs_mapping_file, "AircraftIAInputsMapping", control_mode, start_state)
        self.set_control_mode(control_mode)

        self.IA_commands_labels = ["IA_COM_IDLE", "IA_COM_LIFTOFF", "IA_COM_FIGHT", "IA_COM_RETURN_TO_BASE", "IA_COM_LANDING"]

        self.flag_IA_start_liftoff = True
        self.IA_liftoff_delay = 0
        self.IA_fire_missiles_delay = 10
        self.IA_target_distance_fight = 3000  # Set altitude to target altitude under this distance
        self.IA_fire_missile_cptr = 0
        self.IA_flag_altitude_correction = False
        self.IA_flag_position_correction = False
        self.IA_position_correction_heading = 0
        self.IA_flag_speed_correction = False
        self.IA_flag_go_to_target = False
        self.IA_altitude_min = 500
        self.IA_altitude_max = 8000
        self.IA_altitude_safe = 1500
        self.IA_gun_distance_max = 1000
        self.IA_gun_angle = 10
        self.IA_cruising_altitude = 500
        self.IA_command = AircraftIAControlDevice.IA_COM_IDLE

        self.IA_target_point = hg.Vec3(0, 0, 0)
        self.IA_trajectory_fit_distance = 1000
        self.IA_landing_target = None
        self.IA_flag_landing_target_found = False
        self.IA_flag_goto_landing_approach_point = False
        self.IA_flag_reached_landing_point = False

    def set_control_mode(self, cmode):
        self.control_mode = cmode
        if cmode == ControlDevice.CM_KEYBOARD:
            self.commands.update({
                "ACTIVATE_USER_CONTROL": self.activate_user_control_kb
            })

        elif cmode == ControlDevice.CM_GAMEPAD:
            self.commands.update({
                "ACTIVATE_USER_CONTROL": self.activate_user_control_gp
            })

        elif cmode == ControlDevice.CM_MOUSE:
            self.commands.update({})

    # ==================================== functions

    def activate(self):
        if not self.activated:
            ControlDevice.activate(self)
            aircraft = self.machine
            if not aircraft.wreck:

                td = aircraft.get_device("TargettingDevice")
                if td.target_id == 0:
                    td.search_target()

                n = aircraft.get_machinegun_count()
                for i in range(n):
                    mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                    if mgd is not None and mgd.is_gun_activated():
                        mgd.stop_machine_gun()

                self.IA_flag_go_to_target = False
                if aircraft.flag_landed:
                    if self.IA_liftoff_delay <= 0:
                        self.IA_liftoff_delay = 1
                    self.IA_command = AircraftIAControlDevice.IA_COM_LIFTOFF
                else:
                    if td.target_id == 0:
                        self.IA_command = AircraftIAControlDevice.IA_COM_LANDING
                    else:
                        self.IA_command = AircraftIAControlDevice.IA_COM_FIGHT
                ap_ctrl = aircraft.get_device("AutopilotControlDevice")
                if ap_ctrl is not None:
                    ap_ctrl.activate()
                    ap_ctrl.deactivate_user_control()
                    ap_ctrl.set_autopilot_altitude(aircraft.get_altitude())

    def deactivate(self):
        if self.activated:
            ControlDevice.deactivate(self)
            aircraft = self.machine
            n = aircraft.get_machinegun_count()
            for i in range(n):
                mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and mgd.is_gun_activated():
                    mgd.stop_machine_gun()
            self.IA_flag_go_to_target = False
            aircraft.set_flaps_level(0)
            self.IA_flag_landing_target_found = False
            ap_ctrl = aircraft.get_device("AutopilotControlDevice")
            if ap_ctrl is not None:
                ap_ctrl.deactivate()
                ap_ctrl.activate_user_control()

    def set_IA_landing_target(self, landing_target: LandingTarget):
        self.IA_landing_target = landing_target

    def calculate_landing_projection(self, aircraft, landing_target: LandingTarget):
        pos = aircraft.parent_node.GetTransform().GetPos()
        o = landing_target.get_landing_position()
        oa = pos - o
        uap = landing_target.get_landing_vector()
        dot = hg.Dot(hg.Vec2(oa.x, oa.z), uap)
        oap = uap * dot
        d = hg.Len(oap)
        if dot < 0: d = -d
        return landing_target.get_position(d)

    def calculate_landing_approach_factor(self, landing_target: LandingTarget, landing_projection: hg.Vec3):
        o = landing_target.get_landing_position()
        v = landing_projection - o
        vh = hg.Vec2(v.x, v.z)
        vl = landing_target.get_landing_vector()
        dist = hg.Len(vh)
        if hg.Dot(vl, vh) < 0:
            dist = -dist
        return dist / landing_target.horizontal_amplitude

    def calculate_landing_target_point(self, aircraft, landing_target, landing_proj):
        fit_distance = aircraft.get_linear_speed() * 0.5 * 3.6
        o = landing_target.get_landing_position()
        o = hg.Vec2(o.x, o.z)
        ap = hg.Vec2(landing_proj.x, landing_proj.z)
        pos = aircraft.parent_node.GetTransform().GetPos()
        a = hg.Vec2(pos.x, pos.z)
        dist = hg.Len(ap - a)
        if dist < fit_distance:
            dx = sqrt(fit_distance * fit_distance - dist * dist)
            tdist = hg.Len(ap - o) - dx
            return landing_target.get_position(tdist)
        else:
            return landing_proj

    def update_controlled_device(self, dts):
        aircraft = self.machine
        if not aircraft.wreck and not aircraft.flag_going_to_takeoff_position:
            if self.IA_command == AircraftIAControlDevice.IA_COM_IDLE:
                self.update_IA_idle(aircraft)
            elif self.IA_command == AircraftIAControlDevice.IA_COM_LIFTOFF:
                self.update_IA_liftoff(aircraft, dts)
            elif self.IA_command == AircraftIAControlDevice.IA_COM_FIGHT:
                self.update_IA_fight(aircraft, dts)
            elif self.IA_command == AircraftIAControlDevice.IA_COM_LANDING:
                self.update_IA_landing(aircraft, dts)

    def update_IA_liftoff(self, aircraft, dts):
        self.IA_flag_landing_target_found = False
        aircraft.set_flaps_level(1)
        if self.flag_IA_start_liftoff:
            self.IA_liftoff_delay -= dts
        if self.IA_liftoff_delay <= 0:
            if aircraft.thrust_level < 1 or not aircraft.post_combustion:
                aircraft.set_brake_level(0)
                aircraft.set_flaps_level(1)
                aircraft.set_thrust_level(1)
                aircraft.activate_post_combustion()
                aircraft.angular_levels_dest.z = 0
                aircraft.angular_levels_dest.x = radians(-5)
            if aircraft.ground_node_collision is None:
                aircraft.flag_landed = False
                gear = aircraft.get_device("Gear")
                if gear is not None:
                    if gear.activated:
                        gear.deactivate()
                autopilot = aircraft.get_device("AutopilotControlDevice")
                if autopilot is not None:
                    autopilot.set_autopilot_altitude(200)
                    if aircraft.parent_node.GetTransform().GetPos().y > 100:
                        # if abs(vs) > 1:
                        aircraft.set_flaps_level(0)
                        td = aircraft.get_device("TargettingDevice")
                        if td is not None:
                            td.search_target()
                            if td.target_id == 0:
                                self.IA_command = AircraftIAControlDevice.IA_COM_LANDING
                            else:
                                self.IA_command = AircraftIAControlDevice.IA_COM_FIGHT

    def update_IA_idle(self, aircraft):
        autopilot = aircraft.devices["AutopilotControlDevice"]
        if autopilot is not None:
            autopilot.set_autopilot_speed(400 / 3.6)
            n = aircraft.get_machinegun_count()
            for i in range(n):
                mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                if mgd is not None and mgd.is_gun_activated():
                    mgd.stop_machine_gun()
            autopilot.set_autopilot_altitude(self.IA_cruising_altitude)
            autopilot.set_autopilot_heading(0)

            if aircraft.pitch_attitude > 15:
                aircraft.set_thrust_level(1)
                aircraft.activate_post_combustion()
            elif -15 < aircraft.pitch_attitude < 15:
                aircraft.deactivate_post_combustion()
                aircraft.set_thrust_level(1)
            else:
                aircraft.deactivate_post_combustion()
                aircraft.set_thrust_level(0.5)

    def get_nearest_landing_target(self, aircraft):
        distances = []
        pos = aircraft.parent_node.GetTransform().GetPos()
        for lt in aircraft.landing_targets:
            p = lt.get_approach_entry_position()
            distances.append({"landing_target": lt, "distance": hg.Len(p - pos)})
        distances.sort(key=lambda p: p["distance"])
        return distances[0]["landing_target"]

    def update_IA_landing(self, aircraft, dts):
        if "AutopilotControlDevice" in aircraft.devices and aircraft.devices["AutopilotControlDevice"] is not None:
            autopilot = aircraft.devices["AutopilotControlDevice"]
            if not self.IA_flag_landing_target_found:
                n = aircraft.get_machinegun_count()
                for i in range(n):
                    mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                    if mgd is not None and mgd.is_gun_activated():
                        mgd.stop_machine_gun()
                self.IA_landing_target = self.get_nearest_landing_target(aircraft)
                if self.IA_landing_target is not None:
                    self.IA_flag_landing_target_found = True
                    self.IA_flag_goto_landing_approach_point = True
                    self.IA_flag_reached_landing_point = False
            else:
                pos = aircraft.parent_node.GetTransform().GetPos()
                landing_proj = self.calculate_landing_projection(aircraft, self.IA_landing_target)
                landing_f = self.calculate_landing_approach_factor(self.IA_landing_target, landing_proj)
                if self.IA_flag_goto_landing_approach_point:
                    if landing_f > 1:
                        self.IA_flag_goto_landing_approach_point = False
                    else:
                        p = self.IA_landing_target.get_approach_entry_position()
                        v = p - pos
                        if hg.Len(v) < self.IA_trajectory_fit_distance:
                            self.IA_flag_goto_landing_approach_point = False
                        else:
                            autopilot.set_autopilot_heading(aircraft.calculate_heading(hg.Normalize(v * hg.Vec3(1, 0, 1))))
                            autopilot.set_autopilot_altitude(p.y)
                            autopilot.set_autopilot_speed(2000 / 3.6)
                            aircraft.set_flaps_level(0)
                            aircraft.set_brake_level(0)
                else:
                    # 2D Distance to trajectory:
                    lf = (landing_f - 0.3) / (1 - 0.3)  # lf: near approach parameter (0.3 of total approach)
                    # far approach
                    if lf > 0:
                        self.IA_target_point = self.calculate_landing_target_point(aircraft, self.IA_landing_target, landing_proj)
                        v = self.IA_target_point - pos
                        autopilot.set_autopilot_heading(aircraft.calculate_heading(hg.Normalize(v * hg.Vec3(1, 0, 1))))
                        lfq = floor(lf * 10) / 10
                        autopilot.set_autopilot_speed((2000 * lfq + aircraft.landing_max_speed * (1 - lfq)) / 3.6)
                        aircraft.set_flaps_level(pow(1 - lf, 2) * 0.5)
                        autopilot.set_autopilot_altitude(self.IA_target_point.y)
                    # Near approach
                    else:
                        if "Gear" in aircraft.devices and aircraft.devices["Gear"] is not None:
                            gear = aircraft.devices["Gear"]
                            if not gear.activated:
                                gear.activate()
                            o = self.IA_landing_target.get_landing_position()
                        if landing_f > 0:
                            autopilot.set_autopilot_speed(aircraft.minimum_flight_speed / 3.6)
                            lv = aircraft.get_linear_speed() * 3.6
                            vh = self.IA_landing_target.get_landing_vector() * -100
                            v = hg.Vec3(o.x + vh.x, o.y, o.z + vh.y) - pos
                            autopilot.set_autopilot_heading(aircraft.calculate_heading(hg.Normalize(v * hg.Vec3(1, 0, 1))))
                            alt = hg.GetT(aircraft.parent_node.GetTransform().GetWorld()).y
                            f = max(-1, min(1, ((o.y + 4) - alt) / 2))
                            aircraft.set_flaps_level(0.5 + 0.5 * f)
                            if f < 0:
                                autopilot.set_autopilot_altitude(self.IA_target_point.y + 30 * (1 - aircraft.health_level) + 0 * f)
                            else:
                                autopilot.set_autopilot_altitude(self.IA_target_point.y + 60 * (1 - aircraft.health_level) + 30 * f)
                        else:
                            if not self.IA_flag_reached_landing_point:
                                self.IA_flag_reached_landing_point = True
                                if pos.y < o.y or pos.y > o.y + 10:
                                    self.IA_flag_landing_target_found = False
                            else:
                                autopilot.set_autopilot_speed(-1)
                                aircraft.set_brake_level(1)
                                aircraft.set_thrust_level(0)
                                aircraft.set_flaps_level(0)
                                if aircraft.ground_node_collision is not None:
                                    hs, vs = aircraft.get_world_speed()
                                    if hs < 1 and abs(vs) < 1:
                                        aircraft.set_landed()
                                        self.IA_liftoff_delay = 2
                                        self.IA_command = AircraftIAControlDevice.IA_COM_LIFTOFF

    def update_IA_fight(self, aircraft, dts):
        autopilot = aircraft.devices["AutopilotControlDevice"]
        if autopilot is not None:
            if "Gear" in aircraft.devices and aircraft.devices["Gear"] is not None:
                gear = aircraft.devices["Gear"]
                if gear.activated:
                    gear.deactivate()
            autopilot.set_autopilot_speed(-1)
            speed = aircraft.get_linear_speed() * 3.6  # convert to km/h
            aircraft.set_brake_level(0)
            if speed < aircraft.minimum_flight_speed:
                if not self.IA_flag_speed_correction:
                    self.IA_flag_speed_correction = True
                    aircraft.set_flaps_level(1)
                    aircraft.set_thrust_level(1)
                    aircraft.activate_post_combustion()
                    autopilot.set_autopilot_altitude(aircraft.get_altitude())
                    autopilot.set_autopilot_heading(aircraft.heading)
            else:
                self.IA_flag_speed_correction = False
                aircraft.set_flaps_level(0)
                alt = aircraft.get_altitude()
                td = aircraft.get_device("TargettingDevice")
                if td.target_id > 0:
                    if self.IA_flag_position_correction:
                        if aircraft.playfield_distance < aircraft.playfield_safe_distance / 2:
                            self.IA_flag_position_correction = False

                    elif self.IA_flag_altitude_correction:
                        self.IA_flag_go_to_target = False
                        autopilot.set_autopilot_altitude(self.IA_altitude_safe)
                        if self.IA_altitude_safe - 100 < alt < self.IA_altitude_safe + 100:
                            self.IA_flag_altitude_correction = False

                    else:
                        target_distance = hg.Len(td.targets[td.target_id - 1].get_parent_node().GetTransform().GetPos() - aircraft.parent_node.GetTransform().GetPos())
                        autopilot.set_autopilot_heading(td.target_heading)
                        if target_distance < self.IA_target_distance_fight:
                            self.IA_flag_go_to_target = False
                            autopilot.set_autopilot_altitude(td.target_altitude)
                        else:
                            if not self.IA_flag_go_to_target:
                                self.IA_flag_go_to_target = True
                                aircraft.set_thrust_level(1)
                                aircraft.activate_post_combustion()
                            autopilot.set_autopilot_altitude((td.target_altitude - alt) / 10 + alt)
                        if aircraft.playfield_distance > aircraft.playfield_safe_distance:
                            v = aircraft.parent_node.GetTransform().GetPos() * -1
                            self.IA_position_correction_heading = aircraft.calculate_heading(hg.Normalize(v * hg.Vec3(1, 0, 1)))
                            autopilot.set_autopilot_heading(self.IA_position_correction_heading)
                            self.IA_flag_position_correction = True

                        if alt < self.IA_altitude_min or alt > self.IA_altitude_max:
                            self.IA_flag_altitude_correction = True

                    md = aircraft.get_device("MissilesDevice")

                    if td.target_locked:
                        if md is not None:
                            if self.IA_fire_missile_cptr <= 0:
                                md.fire_missile()
                                self.IA_fire_missile_cptr = self.IA_fire_missiles_delay
                            else:
                                self.IA_fire_missile_cptr -= dts

                    if td.target_angle < self.IA_gun_angle and td.target_distance < self.IA_gun_distance_max:
                        n = aircraft.get_machinegun_count()
                        for i in range(n):
                            mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                            if mgd is not None and not mgd.is_gun_activated():
                                mgd.fire_machine_gun()
                    else:
                        n = aircraft.get_machinegun_count()
                        for i in range(n):
                            mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                            if mgd is not None and mgd.is_gun_activated():
                                mgd.stop_machine_gun()

                    flag_missiles_ok = False
                    if md is not None:
                        for missile in md.missiles:
                            if missile is not None:
                                flag_missiles_ok = True
                    else:
                        flag_missiles_ok = True

                    if not flag_missiles_ok or aircraft.get_num_bullets() == 200 or aircraft.health_level < 0.33:
                        self.IA_flag_landing_target_found = False
                        self.IA_command = AircraftIAControlDevice.IA_COM_LANDING
                        return

                else:
                    self.IA_flag_go_to_target = False
                    n = aircraft.get_machinegun_count()
                    for i in range(n):
                        mgd = aircraft.get_device("MachineGunDevice_%02d" % i)
                        if mgd is not None and mgd.is_gun_activated():
                            mgd.stop_machine_gun()
                    self.IA_flag_landing_target_found = False
                    self.IA_command = AircraftIAControlDevice.IA_COM_LANDING
                    # self.set_autopilot_altitude(self.IA_cruising_altitude)
                    # self.set_autopilot_heading(0)
                # self.stop_machine_gun()

                if not self.IA_flag_go_to_target:
                    if aircraft.pitch_attitude > 15:
                        aircraft.set_thrust_level(1)
                        aircraft.activate_post_combustion()
                    elif -15 < aircraft.pitch_attitude < 15:
                        aircraft.deactivate_post_combustion()
                        aircraft.set_thrust_level(1)
                    else:
                        aircraft.deactivate_post_combustion()
                        aircraft.set_thrust_level(0.5)

    def controlled_device_hitted(self):
        aircraft = self.machine
        td = aircraft.get_device("TargettingDevice")
        offenders = []
        for target_id, target in enumerate(td.targets):
            td_t = target.get_device("TargettingDevice")
            if td_t is not None:
                offender_target = td_t.get_target()
                if offender_target == aircraft:
                    offenders.append([target_id, hg.Len(target.get_parent_node().GetTransform().GetPos() - aircraft.parent_node.GetTransform().GetPos())])
        if len(offenders) > 0:
            if len(offenders) > 1:
                offenders.sort(key=lambda p: p[1])
            td.set_target_id(offenders[0][0])
    # =============================== Keyboard commands ====================================

    def activate_user_control_kb(self, value):
        if ControlDevice.keyboard.Pressed(value):
            uctrl = self.machine.get_device("UserControlDevice")
            if uctrl is not None:
                self.deactivate()
                uctrl.activate()

    # =============================== Gamepad commands ====================================

    def activate_user_control_gp(self, value):
        if ControlDevice.gamepad.Pressed(value):
            uctrl = self.machine.get_device("UserControlDevice")
            if uctrl is not None:
                self.deactivate()
                uctrl.activate()

    # ====================================================================================

    def update_cm_keyboard(self, dts):
        im = self.inputs_mapping["AircraftIAInputsMapping"]["Keyboard"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_gamepad(self, dts):
        im = self.inputs_mapping["AircraftIAInputsMapping"]["GamePad"]
        for cmd, input_code in im.items():
            if cmd in self.commands and input_code != "":
                self.commands[cmd](input_code)

    def update_cm_mouse(self, dts):
        im = self.inputs_mapping["AircraftIAInputsMapping"]["Mouse"]

    def update(self, dts):
        if self.activated:
            if self.flag_user_control and self.machine.has_focus():
                if self.control_mode == ControlDevice.CM_KEYBOARD:
                    self.update_cm_keyboard(dts)
                elif self.control_mode == ControlDevice.CM_GAMEPAD:
                    self.update_cm_gamepad(dts)
                elif self.control_mode == ControlDevice.CM_MOUSE:
                    self.update_cm_mouse(dts)

            self.update_controlled_device(dts)



#============================================================
#   Collision object
#============================================================

class Collisions_Object(MachineDevice):

    _instances = []

    @classmethod
    def reset_collisions_objects(cls):
        cls._instances = []

    @classmethod
    def get_object_by_collision_node(cls, node: hg.Node):
        #nm0 = node.GetName()
        for col_obj in cls._instances:
            nds = col_obj.get_collision_nodes()
            if len(nds) > 0:
                for nd in nds:
                    #nm = nd.GetName()
                    if node.GetUid() == nd.GetUid():
                        return col_obj
        return None

    def __init__(self, name):
        MachineDevice.__init__(self, name, None, True)
        self.collision_nodes = []
        Collisions_Object._instances.append(self)
        self.instance_id = len(Collisions_Object._instances) - 1

    def get_collision_nodes(self):
        return self.collision_nodes

    def test_collision(self, nd: hg.Node):
        if len(self.collision_nodes) > 0:
            for ndt in self.collision_nodes:
                if nd == ndt: return True
        return False


# =====================================================================================================
#                                  Animated model
# =====================================================================================================
class AnimatedModel(Collisions_Object):

    def __init__(self, name, model_name, scene, pipeline_ressource: hg.PipelineResources, instance_scene_name):
        Collisions_Object.__init__(self, name)
        self.commands.update({"SET_CURRENT_PILOT": self.set_current_pilot})
        self.model_name = model_name
        self.scene = scene
        self.res = pipeline_ressource

        self.parent_node, f = hg.CreateInstanceFromAssets(scene, hg.TranslationMat4(hg.Vec3(0, 0, 0)), instance_scene_name, pipeline_ressource, hg.GetForwardPipelineInfo())

        self.parent_node.SetName(self.name)
        self.remove_dummies_objects()

        # Pilots for cocpit view
        self.pilots = self.get_pilots_slots()
        self.current_pilot = 0
        self.set_current_pilot(1)

        # Mobile parts:
        self.parts = {}

    def destroy_particles(self):
        pass

    def setup_particles(self):
        pass

    def reset(self):
        Collisions_Object.reset(self)
        self.set_current_pilot(1)

    def show_objects(self):
        sv = self.parent_node.GetInstanceSceneView()
        nodes = sv.GetNodes(self.scene)
        for i in range(nodes.size()):
            nd = nodes.at(i)
            if nd.HasObject():
                nd.Enable()

    def hide_objects(self):
        sv = self.parent_node.GetInstanceSceneView()
        nodes = sv.GetNodes(self.scene)
        for i in range(nodes.size()):
            nd = nodes.at(i)
            if nd.HasObject():
                nd.Disable()

    def destroy_nodes(self):
        self.scene.DestroyNode(self.parent_node)
        self.parent_node = None
        hg.SceneGarbageCollectSystems(self.scene)

    def get_slots(self, slots_name):
        slots_names = []
        scene_view = self.parent_node.GetInstanceSceneView()
        nodes = scene_view.GetNodes(self.scene)
        n = nodes.size()
        if n == 0:
            raise OSError("ERROR - Empty Instance '" + self.name + "'- Unloaded scene ?")
        for i in range(nodes.size()):
            nd = nodes.at(i)
            nm = nd.GetName()
            if slots_name in nm:
                slots_names.append(nm)
        if len(slots_name) == 0:
            return None
        slots_names.sort()
        slots = []
        for nm in slots_names:
            slots.append(scene_view.GetNode(self.scene, nm))
        return slots

    #  -------- Pilots handler --------------------

    def set_current_pilot(self, pilot_id):
        np = len(self.pilots)
        if np == 0:
            pilot_id = 0
        elif pilot_id > np:
            pilot_id = min(1, len(self.pilots))
        self.current_pilot = pilot_id

    def update_vr_head(self, pilot_id, vr_head_origin: hg.Mat4):
        if pilot_id == 0:
            return
        pilot_id = min(pilot_id, len(self.pilots))
        head_node = self.pilots[pilot_id - 1]["head"]
        v = self.pilots[pilot_id - 1]["nativ_head_position"] - hg.GetT(vr_head_origin)
        head_node.GetTransform().SetPos(v)

    def get_current_pilot_head(self):
        if self.pilots is None or len(self.pilots) == 0:
            return None
        else:
            if self.current_pilot == 0:
                return self.parent_node
            else:
                return self.pilots[self.current_pilot - 1]["head"]

    def get_pilots_slots(self):
        pilots_bodies = self.get_slots("pilote_body")
        heads = self.get_slots("pilote_head")
        pilots = []
        for i in range(len(pilots_bodies)):
            head_pos0 = heads[i].GetTransform().GetPos()
            pilots.append({"body": pilots_bodies[i], "head": heads[i], "nativ_head_position": head_pos0})
        return pilots

    def get_pilots(self):
        return self.pilots

    # ----------- Animations handler----------------------

    def get_animation(self, animation_name):
        return self.parent_node.GetInstanceSceneAnim(animation_name)

    # ---------------------------------------------------

    def get_position(self):
        return self.parent_node.GetTransform().GetPos()

    def get_Euler(self):
        return self.parent_node.GetTransform().GetRot()

    def reset_matrix(self, pos, rot):
        self.parent_node.GetTransform().SetPos(pos)
        self.parent_node.GetTransform().SetRot(rot)
        mat = hg.TransformationMat4(pos, rot)
        self.parent_node.GetTransform().SetWorld(mat)

    def decompose_matrix(self, matrix=None):
        if matrix is None:
            matrix = self.parent_node.GetTransform().GetWorld()
        aX = hg.GetX(matrix)
        aY = hg.GetY(matrix)
        aZ = hg.GetZ(matrix)
        pos = hg.GetT(matrix)
        rot = hg.GetR(matrix)
        return matrix, pos, rot, aX, aY, aZ

    def get_X_axis(self):
        return hg.GetX((self.parent_node.GetTransform().GetWorld()))

    def get_Y_axis(self):
        return hg.GetY((self.parent_node.GetTransform().GetWorld()))

    def get_Z_axis(self):
        return hg.GetZ((self.parent_node.GetTransform().GetWorld()))

    def define_mobile_parts(self, mobile_parts_definitions):
        for mpd in mobile_parts_definitions:
            self.add_mobile_part(mpd[0], mpd[1], mpd[2], mpd[3], mpd[4], mpd[5])

    def add_mobile_part(self, name, angle_min, angle_max, angle_0, node_name, rotation_axis_id):
        sv = self.parent_node.GetInstanceSceneView()
        node = sv.GetNode(self.scene, node_name)
        if angle_0 is None:
            rot = node.GetTransform().GetRot()
            if rotation_axis_id == "X":
                angle_0 = degrees(rot.x)
            elif rotation_axis_id == "Y":
                angle_0 = degrees(rot.y)
            elif rotation_axis_id == "Z":
                angle_0 = degrees(rot.z)
            angle_min += angle_0
            angle_max += angle_0
        self.parts[name] = {"angle_min": angle_min, "angle_max": angle_max, "angle_0": angle_0, "level": 0, "node": node, "part_axis": rotation_axis_id}

    def get_mobile_parts(self):
        return self.parts

    def remove_objects_by_name_pattern(self, name_pattern):
        sv = self.parent_node.GetInstanceSceneView()
        nodes = sv.GetNodes(self.scene)
        for i in range(nodes.size()):
            nd = nodes.at(i)
            nm = nd.GetName()
            if name_pattern in nm:
                if nd.HasObject():
                    nd.RemoveObject()

    def remove_collision_boxes_objects(self):
        self.remove_objects_by_name_pattern("col_shape")

    def remove_dummies_objects(self):
        # RATIONALISER #
        self.remove_objects_by_name_pattern("dummy")
        self.remove_objects_by_name_pattern("slot")
        self.remove_objects_by_name_pattern("pilot_head")
        self.remove_objects_by_name_pattern("pilot_body")

    def get_parent_node(self):
        return self.parent_node

    def enable_nodes(self):
        nodes = self.parent_node.GetInstanceSceneView().GetNodes(self.scene)
        for i in range(nodes.size()):
            nodes.at(i).Enable()

    def disable_nodes(self):
        nodes = self.parent_node.GetInstanceSceneView().GetNodes(self.scene)
        for i in range(nodes.size()):
            nodes.at(i).Disable()

    def copy_mobile_parts_levels(self, src_parts):
        for lbl in self.parts.keys():
            if lbl in src_parts:
                part_main = src_parts[lbl]
                part = self.parts[lbl]
                part["level"] = part_main["level"]

    def update_mobile_parts(self, dts):
        for part in self.parts.values():
            if part["level"] < 0:
                angle = part["angle_0"] * (1 + part["level"]) + part["angle_min"] * -part["level"]
            else:
                angle = part["angle_0"] * (1 - part["level"]) + part["angle_max"] * part["level"]

            trans = part["node"].GetTransform()
            rot = trans.GetRot()

            if part["part_axis"] == "X":
                rot.x = radians(angle)
            elif part["part_axis"] == "Y":
                rot.y = radians(angle)
            elif part["part_axis"] == "Z":
                rot.z = radians(angle)

            trans.SetRot(rot)


# =====================================================================================================
#                                  Destroyable machine
# =====================================================================================================

class Destroyable_Machine(AnimatedModel):

    TYPE_GROUND = 0
    TYPE_SHIP = 1
    TYPE_AIRCRAFT = 2
    TYPE_MISSILE = 3
    TYPE_LANDVEHICLE = 4
    TYPE_MISSILE_LAUNCHER = 5


    flag_activate_particles = True
    flag_update_particles = True
    playfield_max_distance = 50000
    playfield_safe_distance = 40000

    types_labels = ["GROUND", "SHIP", "AICRAFT", "MISSILE", "LANDVEHICLE", "MISSILE_LAUNCHER"]
    update_list = []
    machines_list = []
    machines_items = {}

    @classmethod
    def get_machine_by_node(cls, node:hg.Node):
        collision_object = cls.get_object_by_collision_node(node)
        if collision_object.name in cls.machines_items:
            return cls.machines_items[collision_object.name]
        else:
            return None

    @classmethod
    def reset_machines(cls):
        Collisions_Object.reset_collisions_objects()
        cls.update_list = []

    @classmethod
    def set_activate_particles(cls, flag):
        if flag != cls.flag_activate_particles:
            if flag:
                for machine in cls._instances:
                    machine.setup_particles()
            else:
                for machine in cls._instances:
                    machine.destroy_particles()
        cls.flag_activate_particles = flag

    def __init__(self, name, model_name, scene: hg.Scene, scene_physics, pipeline_ressource: hg.PipelineResources, instance_scene_name, type, nationality, start_position=None, start_rotation=None):

        AnimatedModel.__init__(self, name, model_name, scene, pipeline_ressource, instance_scene_name)

        self.flag_focus = False

        self.playfield_distance = 0

        self.commands.update({"SET_HEALTH_LEVEL": self.set_health_level})

        self.start_position = start_position
        self.start_rotation = start_rotation

        self.flag_custom_physics_mode = False
        self.custom_matrix = None
        self.custom_v_move = None

        self.flag_display_linear_speed = False
        self.flag_display_vertical_speed = False
        self.flag_display_horizontal_speed = False

        self.terrain_altitude = 0
        self.terrain_normale = None

        self.type = type

        self.scene_physics = scene_physics
        self.flag_destroyed = False
        self.flag_crashed = False
        self.nationality = nationality

        self.health_level = 1
        self.wreck = False
        self.v_move = hg.Vec3(0, 0, 0)

        # Linear acceleration:
        self.linear_acceleration = 0
        self.linear_speeds = [0] * 10
        self.linear_spd_rec_cnt = 0

        # Physic Wakeup check:
        self.pos_prec = hg.Vec3(0, 0, 0)
        self.rot_prec = hg.Vec3(0, 0, 0)
        self.flag_moving = False

        self.bottom_height = 1

        # Vertex model:
        self.vs_decl = hg.VertexLayout()
        self.vs_decl.Begin()
        self.vs_decl.Add(hg.A_Position, 3, hg.AT_Float)
        self.vs_decl.Add(hg.A_Normal, 3, hg.AT_Uint8, True, True)
        self.vs_decl.End()
        self.ground_node_collision = None
        # Used by spatialized sound:
        self.view_v_move = hg.Vec3(0, 0, 0)
        self.mat_view = None
        self.mat_view_prec = None

        # Views parameters
        self.camera_track_distance = 40
        self.camera_follow_distance = 40

        # Devices
        self.devices = {}

        # Views parameters
        self.camera_track_distance = 40
        self.camera_follow_distance = 40
        self.camera_tactical_distance = 40
        self.camera_tactical_min_altitude = 10

        # Bounds positions used by collisions raycasts:
        self.collision_boxes = []
        self.bounding_boxe = None
        self.bound_front = hg.Vec3(0, 0, 0)
        self.bound_back = hg.Vec3(0, 0, 0)
        self.bound_up = hg.Vec3(0, 0, 0)
        self.bound_down = hg.Vec3(0, 0, 0)
        self.bound_left = hg.Vec3(0, 0, 0)
        self.bound_right = hg.Vec3(0, 0, 0)

    #============= Devices

    def add_device(self, device: MachineDevice):
        self.devices[device.name] = device

    def remove_device(self, device_name):
        if device_name in self.devices:
            return self.devices.pop(device_name)
        return None

    def update_devices(self, dts):
        for name, device in self.devices.items():
            device.update(dts)

    def get_device(self, device_name):
        if device_name in self.devices:
            return self.devices[device_name]
        else:
            return None

    #==============

    def set_focus(self):
        for machine in self.machines_list:
            if machine.flag_focus:
                machine.flag_focus = False
                break
        self.flag_focus = True

    def has_focus(self):
        return self.flag_focus

    #==============

    def reset(self, position=None, rotation=None):
        AnimatedModel.reset(self)
        if position is not None:
            self.start_position = position
        if rotation is not None:
            self.start_rotation = rotation

        self.set_custom_physics_mode(False)

        self.reset_matrix(self.start_position, self.start_rotation)

    def add_to_update_list(self):
        if self not in Destroyable_Machine.update_list:
            Destroyable_Machine.update_list.append(self)

    def remove_from_update_list(self):
        for i in range(len(Destroyable_Machine.update_list)):
            dm = Destroyable_Machine.update_list
            if dm == self:
                Destroyable_Machine.update_list.pop(i)
                break

    def update_view_v_move(self, dts):
        if self.mat_view_prec is None or self.mat_view is None:
            self.view_v_move.x, self.view_v_move.y, self.view_v_move.z = 0, 0, 0
        else:
            self.view_v_move = (hg.GetT(self.mat_view) - hg.GetT(self.mat_view_prec)) / dts

    def calculate_view_matrix(self, camera):
        cam_mat = camera.GetTransform().GetWorld()
        cam_mat_view = hg.InverseFast(cam_mat)
        self.mat_view_prec = self.mat_view
        self.mat_view = cam_mat_view * self.parent_node.GetTransform().GetWorld()

    def hit(self, value):
        self.set_health_level(self.health_level - value)

    def destroy_nodes(self):
        AnimatedModel.destroy_nodes(self)
        for nd in self.collision_nodes:
            self.scene.DestroyNode(nd)
        self.collision_nodes = []
        hg.SceneGarbageCollectSystems(self.scene, self.scene_physics)

    def setup_collisions(self):

        self.scene.Update(1000)
        self.collision_nodes = []
        self.collision_boxes = []

        nodes = self.parent_node.GetInstanceSceneView().GetNodes(self.scene)
        n = nodes.size()
        for i in range(n):
            nd = nodes.at(i)
            nm = nd.GetName()
            if "col_shape" in nm:
                f, mm = nd.GetObject().GetMinMax(self.res)
                size = (mm.mx - mm.mn)
                mdl = hg.CreateCubeModel(self.vs_decl, size.x, size.y, size.z)
                ref = self.res.AddModel('col_shape' + str(i), mdl)
                pos = nd.GetTransform().GetPos()
                rot = nd.GetTransform().GetRot()
                parent = nd.GetTransform().GetParent()
                material = nd.GetObject().GetMaterial(0)
                new_node_local_matrix = hg.TransformationMat4(pos, rot)
                new_node = hg.CreatePhysicCube(self.scene, hg.Vec3(size), new_node_local_matrix, ref, [material], 0)
                new_node.SetName(self.name + "_ColBox")
                new_node.GetRigidBody().SetType(hg.RBT_Kinematic)
                new_node.GetTransform().SetParent(parent)
                new_node.RemoveObject()
                self.scene.DestroyNode(nd)
                self.collision_nodes.append(new_node)
                self.collision_boxes.append({"node": new_node, "size": hg.Vec3(size)})
                self.scene_physics.NodeCreatePhysicsFromAssets(new_node)
        hg.SceneGarbageCollectSystems(self.scene, self.scene_physics)

    def setup_bounds_positions(self):
        self.scene.Update(1000)
        if len(self.collision_boxes) == 0:
            # Bounds from geometries
            self.setup_objects_bounds_positions()
        else:
            # Bounds from collision shapes
            mdl_mat = hg.InverseFast(self.parent_node.GetTransform().GetWorld())
            bounds = None

            for cb in self.collision_boxes:
                nd = cb["node"]
                mat = mdl_mat * nd.GetTransform().GetWorld()
                size = cb["size"] * 0.5
                bounding_box = [hg.Vec3(-size.x, -size.y, -size.z), hg.Vec3(-size.x, size.y, -size.z), hg.Vec3(size.x, size.y, -size.z), hg.Vec3(size.x, -size.y, -size.z),
                                hg.Vec3(-size.x, -size.y, size.z), hg.Vec3(-size.x, size.y, size.z), hg.Vec3(size.x, size.y, size.z), hg.Vec3(size.x, -size.y, size.z)]

                for pt in bounding_box:
                    pt = mat * pt
                    if bounds is None:
                        bounds = [pt.x, pt.x, pt.y, pt.y, pt.z, pt.z]
                    else:
                        def update_bounds(p, bidx):
                            if p < bounds[bidx]:
                                bounds[bidx] = p
                            elif p > bounds[bidx + 1]:
                                bounds[bidx + 1] = p

                        update_bounds(pt.x, 0)
                        update_bounds(pt.y, 2)
                        update_bounds(pt.z, 4)

            if bounds is not None:
                self.bounding_boxe = [hg.Vec3(bounds[0], bounds[2], bounds[4]), hg.Vec3(bounds[0], bounds[3], bounds[4]), hg.Vec3(bounds[1], bounds[3], bounds[4]), hg.Vec3(bounds[1], bounds[2], bounds[4]),
                                      hg.Vec3(bounds[0], bounds[2], bounds[5]), hg.Vec3(bounds[0], bounds[3], bounds[5]), hg.Vec3(bounds[1], bounds[3], bounds[5]), hg.Vec3(bounds[1], bounds[2], bounds[5])]

                def compute_average(id0, id1, id2, id3):
                    return (self.bounding_boxe[id0] + self.bounding_boxe[id1] + self.bounding_boxe[id2] + self.bounding_boxe[id3]) * 0.25

                self.bound_front = compute_average(4, 5, 6, 7)
                self.bound_back = compute_average(0, 1, 2, 3)
                self.bound_up = compute_average(1, 2, 5, 6)
                self.bound_down = compute_average(0, 3, 4, 7)
                self.bound_left = compute_average(0, 1, 4, 5)
                self.bound_right = compute_average(2, 3, 6, 7)

    def setup_objects_bounds_positions(self):
        #self.scene.Update(0)
        nodes = self.parent_node.GetInstanceSceneView().GetNodes(self.scene)
        n = nodes.size()
        mdl_mat = hg.InverseFast(self.parent_node.GetTransform().GetWorld())
        bounds = None

        for i in range(n):
            nd = nodes.at(i)
            if nd.HasObject():
                mat = nd.GetTransform().GetWorld() * mdl_mat
                f, mm = nd.GetObject().GetMinMax(self.res)
                bounding_box = [hg.Vec3(mm.mn.x, mm.mn.y, mm.mn.z), hg.Vec3(mm.mn.x, mm.mx.y, mm.mn.z), hg.Vec3(mm.mx.x, mm.mx.y, mm.mn.z), hg.Vec3(mm.mx.x, mm.mn.y, mm.mn.z),
                                hg.Vec3(mm.mn.x, mm.mn.y, mm.mx.z), hg.Vec3(mm.mn.x, mm.mx.y, mm.mx.z), hg.Vec3(mm.mx.x, mm.mx.y, mm.mx.z), hg.Vec3(mm.mx.x, mm.mn.y, mm.mx.z)]

                for pt in bounding_box:
                    pt = mat * pt
                    if bounds is None:
                        bounds = [pt.x, pt.x, pt.y, pt.y, pt.z, pt.z]
                    else:
                        def update_bounds(p, bidx):
                            if p < bounds[bidx]:
                                bounds[bidx] = p
                            elif p > bounds[bidx + 1]:
                                bounds[bidx + 1] = p

                        update_bounds(pt.x, 0)
                        update_bounds(pt.y, 2)
                        update_bounds(pt.z, 4)
        if bounds is not None:
            self.bounding_boxe = [hg.Vec3(bounds[0], bounds[2], bounds[4]), hg.Vec3(bounds[0], bounds[3], bounds[4]), hg.Vec3(bounds[1], bounds[3], bounds[4]), hg.Vec3(bounds[1], bounds[2], bounds[4]),
                            hg.Vec3(bounds[0], bounds[2], bounds[5]), hg.Vec3(bounds[0], bounds[3], bounds[5]), hg.Vec3(bounds[1], bounds[3], bounds[5]), hg.Vec3(bounds[1], bounds[2], bounds[5])]

            def compute_average(id0, id1, id2, id3):
                return (self.bounding_boxe[id0] + self.bounding_boxe[id1] + self.bounding_boxe[id2] + self.bounding_boxe[id3]) * 0.25

            self.bound_front = compute_average(4, 5, 6, 7)
            self.bound_back = compute_average(0, 1, 2, 3)
            self.bound_up = compute_average(1, 2, 5, 6)
            self.bound_down = compute_average(0, 3, 4, 7)
            self.bound_left = compute_average(0, 1, 4, 5)
            self.bound_right = compute_average(2, 3, 6, 7)

    def get_world_bounding_boxe(self):
        if self.bounding_boxe is not None:
            matrix = self.parent_node.GetTransform().GetWorld()
            bb = []
            for p in self.bounding_boxe:
                bb.append(matrix * p)
            return bb
        return None

    def get_health_level(self):
        return self.health_level

    def set_health_level(self, level):
        if not self.wreck:
            self.health_level = min(1, max(0, level))
        else:
            self.health_level = 0

    def setup_particles(self):
        pass

    def destroy_particles(self):
        pass

    def get_world_speed(self):
        sX = hg.Vec3.Right * (hg.Dot(hg.Vec3.Right, self.v_move))
        sZ = hg.Vec3.Front * (hg.Dot(hg.Vec3.Front, self.v_move))
        vs = hg.Dot(hg.Vec3.Up, self.v_move)
        hs = hg.Len(sX + sZ)
        return hs, vs

    def get_move_vector(self):
        return self.v_move

    def get_linear_speed(self):
        return hg.Len(self.v_move)

    def set_linear_speed(self, value):
        aZ = hg.GetZ(self.parent_node.GetTransform().GetWorld())
        self.v_move = aZ * value

    def get_altitude(self):
        return self.parent_node.GetTransform().GetPos().y

    def get_pitch_attitude(self):
        aZ = hg.GetZ(self.get_parent_node().GetTransform().GetWorld())
        horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
        pitch_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aZ, aZ)))))
        if aZ.y < 0: pitch_attitude *= -1
        return pitch_attitude

    def get_roll_attitude(self):
        matrix = self.get_parent_node().GetTransform().GetWorld()
        aX = hg.GetX(matrix)
        aY = hg.GetY(matrix)
        aZ = hg.GetZ(matrix)
        if aY.y > 0:
            y_dir = 1
        else:
            y_dir = -1
        horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
        horizontal_aX = hg.Cross(hg.Vec3.Up, horizontal_aZ) * y_dir
        roll_attitude = degrees(acos(max(-1, min(1, hg.Dot(horizontal_aX, aX)))))
        if aX.y < 0: roll_attitude *= -1
        return roll_attitude

    def calculate_heading(self, h_dir: hg.Vec3):
        heading = degrees(acos(max(-1, min(1, hg.Dot(h_dir, hg.Vec3.Front)))))
        if h_dir.x < 0: heading = 360 - heading
        return heading

    def get_heading(self):
        aZ = hg.GetZ(self.get_parent_node().GetTransform().GetWorld())
        horizontal_aZ = hg.Normalize(hg.Vec3(aZ.x, 0, aZ.z))
        return self.calculate_heading(horizontal_aZ)

    def set_custom_physics_mode(self, flag: bool):
        if not flag:
            self.custom_matrix = None
            self.custom_v_move = None
        self.flag_custom_physics_mode = flag

    def get_custom_physics_mode(self):
        return self.flag_custom_physics_mode

    def set_custom_kinetics(self, matrix: hg.Mat4, v_move: hg.Vec3):
        self.custom_matrix = matrix
        self.custom_v_move = v_move

    def update_collisions(self, matrix, dts):
        mat, pos, rot, aX, aY, aZ = self.decompose_matrix(matrix)

        collisions_raycasts = [
            {"name": "down", "position": self.bound_down, "direction": hg.Vec3(0, -4, 0)}
        ]
        ray_hits, self.terrain_altitude, self.terrain_normale = update_collisions(mat, self, collisions_raycasts)

        alt = self.terrain_altitude
        bottom_alt = self.bottom_height
        self.ground_node_collision = None

        for collision in ray_hits:
            if collision["name"] == "down":
                if len(collision["hits"]) > 0:
                    hit = collision["hits"][0]
                    self.ground_node_collision = hit.node
                    alt = hit.P.y + bottom_alt

        if self.flag_crashed:
            pos.y = alt
            self.v_move *= pow(0.9, 60 * dts)

        else:
            if pos.y < alt:
                    pos.y += (alt - pos.y) * 0.1
                    if self.v_move.y < 0: self.v_move.y *= pow(0.8, 60 * dts)

        return hg.TransformationMat4(pos, rot)

    def rec_linear_speed(self):
        self.linear_speeds[self.linear_spd_rec_cnt] = hg.Len(self.v_move)
        self.linear_spd_rec_cnt = (self.linear_spd_rec_cnt + 1) % len(self.linear_speeds)

    def update_linear_acceleration(self):
        m = 0
        for s in self.linear_speeds:
            m += s
        m /= len(self.linear_speeds)
        self.linear_acceleration = hg.Len(self.v_move) - m

    def get_linear_acceleration(self):
        return self.linear_acceleration

    def update_feedbacks(self, dts):
        pass

    def update_physics_wakeup(self):
        for nd in self.collision_nodes:
            self.scene_physics.NodeWake(nd)
        return
        trans = self.get_parent_node().GetTransform()
        pos = trans.GetPos()
        rot = trans.GetRot()
        rv = rot - self.rot_prec
        v = hg.Len(pos - self.pos_prec)
        if self.flag_moving:
            if v < 0.1 and abs(rv.x) < 0.01 and abs(rv.y) < 0.01 and abs(rv.z) < 0.01:
                self.flag_moving = False
            else:
                self.pos_prec.x, self.pos_prec.y, self.pos_prec.z = pos.x, pos.y, pos.z
                self.rot_prec.x, self.rot_prec.y, self.rot_prec.z = rot.x, rot.y, rot.z
        else:
            if v > 0.1 or abs(rv.x) > 0.001 or abs(rv.y) > 0.001 or abs(rv.z) > 0.001:
                self.flag_moving = True
                if self.type == Destroyable_Machine.TYPE_AIRCRAFT:
                    print("YOUPLA")
                for nd in self.collision_nodes:
                    self.scene_physics.NodeWake(nd)


    def get_physics_parameters(self):
        return {"v_move": self.v_move,
                "thrust_level": 0,
                "thrust_force": 0,
                "lift_force": 0,
                "drag_coefficients": hg.Vec3(0, 0, 0),
                "health_wreck_factor": 1,
                "angular_levels": hg.Vec3(0, 0, 0),
                "angular_frictions": hg.Vec3(0, 0, 0),
                "speed_ceiling": 0,
                "flag_easy_steering": False
                }

    def update_kinetics(self, dts):
        if self.activated:
            if self.custom_matrix is not None:
                matrix = self.custom_matrix
            else:
                matrix = self.get_parent_node().GetTransform().GetWorld()
            if self.custom_v_move is not None:
                v_move = self.custom_v_move
            else:
                v_move = self.v_move

            if not self.flag_crashed:
                self.v_move = v_move

            # Apply displacement vector and gravity
            if not self.flag_custom_physics_mode:
                self.v_move += F_gravity * dts
                pos = hg.GetT(matrix)
                pos += self.v_move * dts
                hg.SetT(matrix, pos)

            # Collisions
            mat = self.update_collisions(matrix, dts)

            mat, pos, rot, aX, aY, aZ = self.decompose_matrix(mat)

            self.parent_node.GetTransform().SetPos(pos)
            self.parent_node.GetTransform().SetRot(rot)

            self.rec_linear_speed()
            self.update_linear_acceleration()

            self.update_devices(dts)

            self.update_mobile_parts(dts)
            self.update_feedbacks(dts)

    def rearm(self):
        pass

# =====================================================================================================
#                                  Missile
# =====================================================================================================

class Missile(Destroyable_Machine):
    num_smoke_parts = 17

    def __init__(self, name, model_name, nationality, scene: hg.Scene, scene_physics, pipeline_ressource: hg.PipelineResources, instance_scene_name,
                 smoke_color: hg.Color = hg.Color.White, start_position=hg.Vec3.Zero, start_rotation=hg.Vec3.Zero):

        self.flag_user_control = False

        self.angular_levels = hg.Vec3(0, 0, 0)

        self.start_position = start_position
        self.start_rotation = start_rotation
        self.smoke_color = smoke_color
        self.smoke_color_label = "uColor"

        Destroyable_Machine.__init__(self, name, model_name, scene, scene_physics, pipeline_ressource, instance_scene_name, Destroyable_Machine.TYPE_MISSILE, nationality)
        self.commands.update({"SET_ROLL_LEVEL": self.set_roll_level,
                              "SET_PITCH_LEVEL": self.set_pitch_level,
                              "SET_YAW_LEVEL": self.set_yaw_level
                              })

        self.target = None
        self.target_collision_test_distance_max = 100

        # Missile constantes:
        self.thrust_force = 100
        self.smoke_parts_distance = 1.44374
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.angular_frictions = hg.Vec3(0.0001, 0.0001, 0.0001)  # pitch, yaw, roll
        self.life_delay = 20
        self.smoke_delay = 1
        self.speed_ceiling = 4000

        self.smoke_time = 0
        self.life_cptr = 0

        self.engines_slots = self.get_engines_slots()

        # Feed-backs and particles:
        self.smoke = []
        self.explode = None
        if len(self.engines_slots) > 0:
            if Destroyable_Machine.flag_activate_particles:
                self.setup_particles()

        self.flag_armed = True

        self.setup_bounds_positions()

        # UserControlDevice used for debugging
        self.add_device(MissileUserControlDevice("MissileUserControlDevice", self))

    def set_roll_level(self, value):
        pass

    def set_pitch_level(self, value):
        pass

    def set_yaw_level(self, value):
        pass

    def get_valid_targets_list(self):
        targets = []
        for machine in Destroyable_Machine.machines_list:
            if machine.nationality != self.nationality and (machine.type == Destroyable_Machine.TYPE_AIRCRAFT):
                targets.append(machine)
        return targets

    def set_life_delay(self, life_delay):
        self.life_delay = life_delay

    def is_armed(self):
        return self.flag_armed

    def destroy(self):
        if not self.flag_destroyed:
            self.get_parent_node().GetTransform().ClearParent()
            self.destroy_particles()
            self.destroy_nodes()
            self.flag_destroyed = True
            self.remove_from_update_list()
        # scene.GarbageCollect()

    def setup_particles(self):
        for i in range(Missile.num_smoke_parts):
            node = duplicate_node_object(self.scene, self.scene.GetNode("enemymissile_smoke" + "." + str(i)), self.name + ".smoke_" + str(i))
            self.smoke.append({"node":node, "alpha": 0})
        self.set_smoke_color(self.smoke_color)

        if self.explode is not None:
            self.destroy_particles()
        self.explode = ParticlesEngine(self.name + ".explode", self.scene, "feed_back_explode",
                                       50, hg.Vec3(5, 5, 5), hg.Vec3(100, 100, 100), 180, 0)
        self.explode.delay_range = hg.Vec2(1, 2)
        self.explode.flow = 0
        self.explode.scale_range = hg.Vec2(0.25, 2)
        self.explode.start_speed_range = hg.Vec2(0, 100)
        self.explode.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0., 0., 0.5), hg.Color(0., 0., 0., 0.25),
                               hg.Color(0., 0., 0., 0.125), hg.Color(0., 0., 0., 0.0)]
        self.explode.set_rot_range(radians(20), radians(50), radians(10), radians(45), radians(5), radians(15))
        self.explode.gravity = hg.Vec3(0, -9.8, 0)
        self.explode.loop = False

    def destroy_particles(self):
        if self.explode is not None:
            self.explode.destroy()
            self.explode = None
        for p in self.smoke:
            self.scene.DestroyNode(p["node"])
            self.scene.GarbageCollect()
        self.smoke = []

    def activate(self):
        if self.explode is not None:
            self.explode.activate()
        for p in self.smoke:
            p["node"].Enable()
        self.enable_nodes()
        Destroyable_Machine.activate(self)

    def deactivate(self):
        # CONFUSION - ADD HIDE() Function
        Destroyable_Machine.deactivate(self)
        self.disable_nodes()
        for p in self.smoke:
            p["node"].Disable()
        if self.explode is not None:
            self.explode.deactivate()
        self.remove_from_update_list()

    def get_engines_slots(self):
        return self.get_slots("engine_slot")

    def reset(self, position=None, rotation=None):

        # Don't call parent's function, World Matrix mustn't be reseted !
        #Destroyable_Machine.reset(self, position, rotation)

        if position is not None:
            self.start_position = position
        if rotation is not None:
            self.start_rotation = rotation

        self.parent_node.GetTransform().SetPos(self.start_position)
        self.parent_node.GetTransform().SetRot(self.start_rotation)

        self.set_custom_physics_mode(False)

        self.smoke_time = 0

        self.remove_from_update_list()
        self.deactivate()
        for p in self.smoke:
            p["node"].GetTransform().SetPos(hg.Vec3(0, 0, 0))
            p["node"].Enable()
        if self.explode is not None:
            self.explode.reset()
            self.explode.flow = 0
        self.enable_nodes()
        self.wreck = False
        self.v_move *= 0
        self.life_cptr = 0

    def set_smoke_color(self, color: hg.Color):
        self.smoke_color = color
        for p in self.smoke:
            p["node"].GetTransform().SetPos(hg.Vec3(0, 0, 0))
            hg.SetMaterialValue(p["node"].GetObject().GetMaterial(0), self.smoke_color_label, hg.Vec4(self.smoke_color.r, self.smoke_color.g, self.smoke_color.b, self.smoke_color.a))

    def get_target_id(self):
        if self.target is not None:
            return self.target.name
        else:
            return ""

    def set_target(self, target: Destroyable_Machine):
        if target is not None and target.nationality != self.nationality:
            self.target = target
        else:
            self.target = None

    def set_target_by_name(self, target_name):
        self.target = None
        if target_name != "":
            for target in Destroyable_Machine.machines_list:
                if target.name == target_name:
                    self.set_target(target)
                    break


    def start(self, target: Destroyable_Machine, v0: hg.Vec3):
        if not self.activated:
            self.smoke_time = 0
            self.life_cptr = 0
            self.set_target(target)
            self.v_move = hg.Vec3(v0)
            self.activated = True
            self.add_to_update_list()
            pos = self.parent_node.GetTransform().GetPos()
            for p in self.smoke:
                p["node"].Enable()
                p["node"].GetTransform().SetPos(pos)

    def update_smoke(self, target_point: hg.Vec3, dts):
        spd = self.get_linear_speed() * 0.033
        t = min(1, abs(self.smoke_time) / self.smoke_delay)
        new_t = self.smoke_time + dts
        if self.smoke_time < 0 and new_t >= 0:
            # pos0=hg.Vec3(0,-1000,0)
            for i in range(len(self.smoke)):
                node = self.smoke[i]["node"]
                node.Disable()  # GetTransform().SetPos(pos0)
            self.smoke_time = new_t
        else:
            self.smoke_time = new_t
            n = len(self.smoke)
            color_end = self.smoke_color * t + hg.Color(1., 1., 1., 0.) * (1 - t)
            for i in range(n):
                node = self.smoke[i]["node"]

                if self.wreck:
                     alpha = self.smoke[i]["alpha"] * (1 - i / n) * t
                else:
                    mat = node.GetTransform().GetWorld()
                    hg.SetScale(mat, hg.Vec3(1, 1, 1))
                    pos = hg.GetT(mat)
                    v = target_point - pos
                    smoke_part_spd = hg.Len(v)
                    dir = hg.Normalize(v)
                    # Position:
                    if smoke_part_spd > self.smoke_parts_distance * spd:
                        pos = target_point - dir * self.smoke_parts_distance * spd
                        node.GetTransform().SetPos(hg.Vec3(pos))
                        alpha = color_end.a * (1 - i / n)
                    else:
                        alpha = 0
                    
                    self.smoke[i]["alpha"] = alpha
                    # node.Enable()
                    # else:
                    # node.Disable()
                    # Orientation:
                    aZ = hg.Normalize(hg.GetZ(mat))
                    axis_rot = hg.Cross(aZ, dir)
                    angle = hg.Len(axis_rot)
                    if angle > 0.001:
                        # Rotation matrix:
                        ay = hg.Normalize(axis_rot)
                        rot_mat = hg.Mat3(hg.Cross(ay, dir), ay, dir)
                        node.GetTransform().SetRot(hg.ToEuler(rot_mat))
                    node.GetTransform().SetScale(hg.Vec3(1, 1, spd))
                    target_point = pos

                hg.SetMaterialValue(node.GetObject().GetMaterial(0), self.smoke_color_label, hg.Vec4(color_end.r, color_end.g, color_end.b, alpha))

               

    def get_hit_damages(self):
        raise NotImplementedError

    def update_collisions(self, matrix, dts):

        smoke_start_pos = hg.GetT(self.engines_slots[0].GetTransform().GetWorld())

        collisions_raycasts = []
        if self.target is not None:
            distance = hg.Len(self.target.get_parent_node().GetTransform().GetPos() - hg.GetT(matrix))
            if distance < self.target_collision_test_distance_max:


                #debug
                """
                if not self.flag_user_control:
                    self.flag_user_control = True
                    ucd = self.get_device("MissileUserControlDevice")
                    if ucd is not None:
                        ucd.activate()
                        ucd.pos_mem = hg.GetT(matrix)
                """

                #Ajouter des slots pour les points de départ des raycasts
                raycast_length = hg.Len(self.v_move) #50
                collisions_raycasts.append({"name": "front", "position": self.bound_front + hg.Vec3(0, 0, 0.4), "direction": hg.Vec3(0, 0, raycast_length)}) #)})

            """
            else:
                # debug
                if self.flag_user_control:
                    self.flag_user_control = False
                    ucd = self.get_device("MissileUserControlDevice")
                    if ucd is not None:
                        ucd.deactivate()
            """



        rays_hits, self.terrain_altitude, self.terrain_normale = update_collisions(matrix, self, collisions_raycasts)

        pos = hg.GetT(matrix)
        rot = hg.GetR(matrix)
        self.parent_node.GetTransform().SetRot(rot)
        # self.v_move = physics_parameters["v_move"]

        # Collision
        if self.target is not None:

            for collision in rays_hits:
                if collision["name"] == "front":
                    if len(collision["hits"]) > 0:
                        hit = collision["hits"][0]
                        if 0 < hit.t < raycast_length:
                            v_impact = hit.P - pos
                            if hg.Len(v_impact) < 2 * hg.Len(self.v_move) * dts:
                                collision_object = Collisions_Object.get_object_by_collision_node(hit.node)
                                if collision_object is not None and hasattr(collision_object, "nationality") and collision_object.nationality != self.nationality:
                                    self.start_explosion()
                                    collision_object.hit(self.get_hit_damages())

        #debug:
        if self.flag_user_control:
            ucd = self.get_device("MissileUserControlDevice")
            if ucd is not None:
                self.parent_node.GetTransform().SetPos(ucd.pos_mem)

        else:
            self.parent_node.GetTransform().SetPos(pos)

        if pos.y < self.terrain_altitude:
            self.start_explosion()
        smoke_start_pos += self.v_move * dts
        self.update_smoke(smoke_start_pos, dts)

    def get_physics_parameters(self):
        return {"v_move": self.v_move,
                "thrust_level": 1,
                "thrust_force": self.thrust_force,
                "lift_force": 0,
                "drag_coefficients": self.drag_coeff,
                "health_wreck_factor": 1,
                "angular_levels": self.angular_levels,
                "angular_frictions": self.angular_frictions,
                "speed_ceiling": self.speed_ceiling,
                "flag_easy_steering": False
                }

    def update_kinetics(self, dts):

        if self.activated:

            self.update_devices(dts)

            self.life_cptr += dts

            if 0 < self.life_delay < self.life_cptr:
                self.start_explosion()
            if not self.wreck:

                if not self.flag_custom_physics_mode:
                    mat, pos, rot, aX, aY, aZ = self.decompose_matrix()
                    # Rotation
                    self.angular_levels.x, self.angular_levels.y, self.angular_levels.z = 0, 0, 0
                    if self.target is not None:
                        target_node = self.target.get_parent_node()
                        target_dir = hg.Normalize((target_node.GetTransform().GetPos() - pos))
                        axis_rot = hg.Cross(aZ, target_dir)
                        if hg.Len(axis_rot) > 0.001:
                            moment = hg.Normalize(axis_rot)
                            self.angular_levels.x = hg.Dot(aX, moment)
                            self.angular_levels.y = hg.Dot(aY, moment)
                            self.angular_levels.z = hg.Dot(aZ, moment)

                    physics_parameters = self.get_physics_parameters()

                    mat, physics_parameters = update_physics(self.parent_node.GetTransform().GetWorld(), self, physics_parameters, dts)
                    self.v_move = physics_parameters["v_move"]

                    #debug
                    if self.flag_user_control:
                        self.v_move *= 0


                else:
                    if self.custom_matrix is not None:
                        mat = self.custom_matrix
                    else:
                        mat = self.get_parent_node().GetTransform().GetWorld()
                    if self.custom_v_move is not None:
                        self.v_move = self.custom_v_move


                self.update_collisions(mat, dts)

            else:
                pos = self.parent_node.GetTransform().GetPos()
                smoke_start_pos = hg.GetT(self.engines_slots[0].GetTransform().GetWorld())

                if self.explode is not None:
                    self.explode.update_kinetics(pos, hg.Vec3.Front, self.v_move, hg.Vec3.Up, dts)
                    if len(self.smoke) > 0:
                        if self.smoke_time < 0:
                            self.update_smoke(smoke_start_pos, dts)
                        if self.explode.end and self.smoke_time >= 0:
                            self.deactivate()
                    elif self.explode.end:
                        self.deactivate()
                else:
                    if len(self.smoke) > 0:
                        if self.smoke_time < 0:
                            self.update_smoke(smoke_start_pos, dts)
                        if self.smoke_time >= 0:
                            self.deactivate()
                    else:
                        self.deactivate()

    def start_explosion(self):
        if not self.wreck:
            self.wreck = True
            if self.explode is not None:
                self.explode.flow = 3000
            self.disable_nodes()
            self.smoke_time = -self.smoke_delay
        # self.parent_node.RemoveObject()

    def get_target_name(self):

        if self.target is None:
            return ""
        else:
            return self.target.name
    
    def set_thrust_force(self, value:float):
        self.thrust_force = value
    
    def set_angular_friction(self, x, y, z):
        self.angular_frictions.x, self.angular_frictions.y, self.angular_frictions.z = x, y, z
    
    def set_drag_coefficients(self, x, y, z):
        self.drag_coeff.x, self.drag_coeff.y, self.drag_coeff.z = x, y, z

# =====================================================================================================
#                                  Aircraft
# =====================================================================================================

class Aircraft(Destroyable_Machine):
    IA_COM_IDLE = 0
    IA_COM_LIFTOFF = 1
    IA_COM_FIGHT = 2
    IA_COM_RETURN_TO_BASE = 3
    IA_COM_LANDING = 4

    def __init__(self, name, model_name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, instance_scene_name, nationality, start_position, start_rotation):

        Destroyable_Machine.__init__(self, name, model_name, scene, scene_physics, pipeline_ressource, instance_scene_name, Destroyable_Machine.TYPE_AIRCRAFT, nationality, start_position, start_rotation)
        self.commands.update({"SET_THRUST_LEVEL": self.set_thrust_level,
                              "SET_BRAKE_LEVEL": self.set_brake_level,
                              "SET_FLAPS_LEVEL": self.set_flaps_level,
                              "SET_ROLL_LEVEL": self.set_roll_level,
                              "SET_PITCH_LEVEL": self.set_pitch_level,
                              "SET_YAW_LEVEL": self.set_yaw_level,
                              "ACTIVATE_POST_COMBUSTION": self.activate_post_combustion,
                              "DEACTIVATE_POST_COMBUSTION": self.deactivate_post_combustion
                              })

        self.add_device(AircraftUserControlDevice("UserControlDevice", self, "scripts/aircraft_user_inputs_mapping.json"))
        self.add_device(AircraftAutopilotControlDevice("AutopilotControlDevice", self, "scripts/aircraft_autopilot_inputs_mapping.json"))
        self.add_device(AircraftIAControlDevice("IAControlDevice", self, "scripts/aircraft_ia_inputs_mapping.json"))
        self.add_device(TargettingDevice("TargettingDevice", self, True))
        self.setup_collisions()

        # Aircraft vars:
        self.post_combustion = False
        self.angular_levels = hg.Vec3(0, 0, 0)  # 0 to 1
        self.brake_level = 0
        self.flag_landing = False
        self.thrust_level = 0

        self.start_landed = True

        self.start_gear_state = True # !!!!!!!!!!!!!!!!!!!!!!!

        self.start_thrust_level = 0
        self.start_linear_speed = 0

        # Setup slots
        self.engines_slots = self.get_engines_slots()

        self.machine_gun_slots = self.get_machines_guns_slots()

        # Missiles:
        self.missiles_slots = self.get_missiles_slots()
        if self.missiles_slots is not None:
            self.add_device(MissilesDevice("MissilesDevice", self, self.missiles_slots))

        """
        self.missiles_slots = self.get_missiles_slots()
        self.missiles = [None] * len(self.missiles_slots)

        self.missiles_started = [None] * len(self.missiles_slots)
        """

        # Gun machine:
        n = len(self.machine_gun_slots)
        for i in range(n):
            self.add_device(MachineGun(("MachineGunDevice_%02d") % i, self, self.machine_gun_slots[i], scene, scene_physics, 1000 / n))
            #self.gun_machines.append(MachineGun(self.name + ".gun." + str(i + 1), self.scene, self.scene_physics, 1000))

        # Particles:
        self.explode = None
        self.smoke = None
        self.post_combustion_particles = []
        if Destroyable_Machine.flag_activate_particles:
            self.setup_particles()


        #self.IA_commands_labels = ["IA_COM_IDLE", "IA_COM_LIFTOFF", "IA_COM_FIGHT", "IA_COM_RETURN_TO_BASE", "IA_COM_LANDING"]

        # Aircraft constants:

        self.landing_max_speed = 300  # km/h
        self.bottom_height = 0.96
        self.gear_height = 2
        self.parts_angles = hg.Vec3(radians(15), radians(45), radians(45))
        self.thrust_force = 10
        self.post_combustion_force = self.thrust_force / 2
        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.gear_drag = 0.001
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
        self.speed_ceiling = 1750  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 20000
        self.max_altitude = 30000
        self.landing_max_speed = 300  # km/h

        # Aircraft vars:
        self.flag_going_to_takeoff_position = False
        self.takeoff_position = None

        self.flag_easy_steering = True
        self.flag_easy_steering_mem = True  # Used in IA on/off switching
        self.thrust_level_inertia = 1
        self.thrust_level_dest = 0
        self.thrust_disfunction_noise = Temporal_Perlin_Noise(0.1)
        self.brake_level_dest = 0
        self.brake_level_inertia = 1
        self.flaps_level = 0
        self.flaps_level_dest = 0
        self.flaps_level_inertia = 1
        self.angular_levels_dest = hg.Vec3(0, 0, 0)

        self.landing_targets = []

        # Attitudes calculation:
        self.pitch_attitude = 0
        self.roll_attitude = 0
        self.heading = 0

        self.flag_landed = True
        self.minimum_flight_speed = 250
        self.reset()

    def reset(self, position=None, rotation=None):
        Destroyable_Machine.reset(self, position, rotation)

        # print("Start_position: " + str(self.start_position.x) + " " + str(self.start_position.y) + " " + str(self.start_position.z))

        self.v_move = hg.GetZ(self.parent_node.GetTransform().GetWorld()) * self.start_linear_speed
        self.angular_levels = hg.Vec3(0, 0, 0)

        if self.smoke is not None: self.smoke.reset()
        if self.explode is not None: self.explode.reset()
        self.wreck = False
        self.flag_crashed = False

        #self.flag_gear_deployed = self.start_gear_state
        if "Gear" in self.devices and self.devices["Gear"] is not None:
            self.devices["Gear"].reset()

        n = self.get_machinegun_count()
        for i in range(n):
            gmd = self.get_device("MachineGunDevice_%02d" % i)
            if gmd is not None:
                gmd.reset()

        td = self.get_device("TargettingDevice")
        if td is not None:
            td.reset()
        self.rearm()

        self.reset_thrust_level(self.start_thrust_level)
        self.reset_brake_level(0)
        self.reset_flaps_level(0)

        self.post_combustion = False

        #self.deactivate_autopilot()
        #self.deactivate_IA()
        ia = self.get_device("IAControlDevice")
        if ia is not None:
            ia.deactivate()
        apctrl = self.get_device("AutopilotControlDevice")
        if apctrl is not None:
            apctrl.deactivate()


        self.set_health_level(1)
        self.angular_levels_dest = hg.Vec3(0, 0, 0)

        self.linear_speeds = [self.start_linear_speed] * 10
        self.linear_acceleration = 0

        self.flag_landed = self.start_landed

    def hit(self, value):
        if not self.wreck:
            self.set_health_level(self.health_level - value)
            if self.health_level == 0 and not self.wreck:
                self.start_explosion()
            ia_ctrl = self.get_device("IAControlDevice")
            if ia_ctrl is not None and ia_ctrl.is_activated():
                ia_ctrl.controlled_device_hitted()

    def show_objects(self):
        AnimatedModel.show_objects(self)
        # A MODIFIER
        self.devices["Gear"].start_state = self.devices["Gear"].activated
        self.devices["Gear"].reset()

    def destroy(self):

        md = self.get_device("MissilesDevice")
        if md is not None:
            md.destroy()

        n = self.get_machinegun_count()
        for i in range(n):
            gmd = self.get_device("MachineGunDevice_%02d" % i)
            if gmd is not None:
                gmd.destroy_gun()

        if self.explode is not None:
            self.explode.destroy()
            self.explode = None
        if self.smoke is not None:
            self.smoke.destroy()
            self.smoke = None

        for pcp in self.post_combustion_particles:
            pcp.destroy()
        self.post_combustion_particles = []
        self.destroy_nodes()
        self.flag_destroyed = True

    def set_health_level(self, value):
        self.health_level = min(max(value, 0), 1)
        if self.smoke is not None:
            if self.health_level < 1:
                self.smoke.flow = int(self.smoke.num_particles / 10)
            else:
                self.smoke.flow = 0
            self.smoke.delay_range = hg.Vec2(1, 10) * pow(1 - self.health_level, 3)
            self.smoke.scale_range = hg.Vec2(0.1, 5) * pow(1 - self.health_level, 3)
            self.smoke.stream_angle = pow(1 - self.health_level, 2.6) * 180

    def setup_bounds_positions(self):
        Destroyable_Machine.setup_bounds_positions(self)
        self.bound_down.y -= 0.4

    # ===================== Start state:

    def record_start_state(self):
        self.start_position = self.parent_node.GetTransform().GetPos()
        self.start_rotation = self.parent_node.GetTransform().GetRot()
        self.devices["Gear"].start_state = self.devices["Gear"].activated
        self.start_landed = self.flag_landed
        self.start_thrust_level = self.thrust_level
        self.start_linear_speed = self.get_linear_speed()

    # ==================== Thrust ===================================

    def get_thrust_level(self):
        return self.thrust_level

    def set_thrust_level(self, value):
        self.thrust_level_dest = min(max(value, 0), 1)

    def reset_thrust_level(self, value):
        self.thrust_level_dest = min(max(value, 0), 1)
        self.thrust_level = self.thrust_level_dest

    def update_thrust_level(self, dts):

        # Altitude disfunctions:
        alt = self.get_altitude()
        collapse = 1
        if alt > self.max_safe_altitude:
            f = pow((alt - self.max_safe_altitude) / (self.max_altitude - self.max_safe_altitude), 2)
            perturb = (self.thrust_disfunction_noise.temporal_Perlin_noise(dts) * 0.5 + 0.5) * f
            collapse = 1 - perturb
            self.hit(self.thrust_level * 0.001 * perturb)

        dest = self.thrust_level_dest * collapse

        if dest > self.thrust_level:
            self.thrust_level = min(dest, self.thrust_level + self.thrust_level_inertia * dts)
        else:
            self.thrust_level = max(dest, self.thrust_level - self.thrust_level_inertia * dts)

        # Clamp:
        self.thrust_level = min(1, max(0, self.thrust_level))

        if self.thrust_level < 1: self.deactivate_post_combustion()

    def activate_post_combustion(self):
        if self.thrust_level == 1:
            self.post_combustion = True
            for pcp in self.post_combustion_particles:
                pcp.flow = 35

    def deactivate_post_combustion(self):
        self.post_combustion = False
        for pcp in self.post_combustion_particles:
            pcp.flow = 0

    # ==================== Brakes ==========================

    def get_brake_level(self):
        return self.brake_level

    def reset_brake_level(self, value):
        self.brake_level_dest = min(max(value, 0), 1)
        self.brake_level = self.brake_level_dest

    def set_brake_level(self, value):
        self.brake_level_dest = min(max(value, 0), 1)

    def update_brake_level(self, dts):
        if self.brake_level_dest > self.brake_level:
            self.brake_level = min(self.brake_level_dest, self.brake_level + self.brake_level_inertia * dts)
        else:
            self.brake_level = max(self.brake_level_dest, self.brake_level - self.brake_level_inertia * dts)

    # ==================== flaps ==========================

    def get_flaps_level(self):
        return self.flaps_level

    def reset_flaps_level(self, value):
        self.flaps_level_dest = min(max(value, 0), 1)
        self.flaps_level = self.flaps_level_dest

    def set_flaps_level(self, value):
        self.flaps_level_dest = min(max(value, 0), 1)

    def update_flaps_level(self, dts):
        if self.flaps_level_dest > self.flaps_level:
            self.flaps_level = min(self.flaps_level_dest, self.flaps_level + self.flaps_level_inertia * dts)
        else:
            self.flaps_level = max(self.flaps_level_dest, self.flaps_level - self.flaps_level_inertia * dts)

    # ==================== Rotations ===============================

    def set_pitch_level(self, value):
        self.angular_levels_dest.x = max(min(1, value), -1)

    def set_yaw_level(self, value):
        self.angular_levels_dest.y = max(min(1, value), -1)

    def set_roll_level(self, value):
        self.angular_levels_dest.z = max(min(1, value), -1)

    def get_pilot_pitch_level(self):
        return self.angular_levels_dest.x

    def get_pilot_yaw_level(self):
        return self.angular_levels_dest.y

    def get_pilot_roll_level(self):
        return self.angular_levels_dest.z

    # ===================== Particles ==============================

    def setup_particles(self):
        # Explode particles:
        self.explode = ParticlesEngine(self.name + ".explode", self.scene, "feed_back_explode", 100, hg.Vec3(10, 10, 10), hg.Vec3(100, 100, 100), 180, 0)
        self.explode.delay_range = hg.Vec2(1, 4)
        self.explode.flow = 0
        self.explode.scale_range = hg.Vec2(0.25, 2)
        self.explode.start_speed_range = hg.Vec2(10, 150)
        self.explode.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 1., 0.9, 0.5), hg.Color(0.6, 0.525, 0.5, 0.25),
                               hg.Color(0.5, 0.5, 0.5, 0.125), hg.Color(0., 0., 0., 0.0)]
        self.explode.set_rot_range(radians(20), radians(150), radians(50), radians(120), radians(45), radians(120))
        self.explode.gravity = hg.Vec3(0, -9.8, 0)
        self.explode.loop = False

        # Smoke particles:
        self.smoke = ParticlesEngine(self.name + ".smoke", self.scene, "feed_back_explode", int(uniform(200, 400)), hg.Vec3(5, 5, 5), hg.Vec3(50, 50, 50), 180, 0)
        self.smoke.flow_decrease_date = 0.5
        self.smoke.delay_range = hg.Vec2(4, 8)
        self.smoke.flow = 0
        self.smoke.scale_range = hg.Vec2(0.1, 5)
        self.smoke.start_speed_range = hg.Vec2(5, 15)
        self.smoke.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0.2, 0.1, 0.3), hg.Color(.7, .7, .7, 0.2),
                             hg.Color(.5, .5, .5, 0.1), hg.Color(0., 0., 0., 0.05), hg.Color(0., 0.5, 1., 0)]
        self.smoke.set_rot_range(0, 0, radians(120), radians(120), 0, 0)
        self.smoke.gravity = hg.Vec3(0, 30, 0)
        self.smoke.linear_damping = 0.5
        self.smoke.loop = True

        # Post-combustion particles:
        for i in range(len(self.engines_slots)):
            self.post_combustion_particles.append(self.create_post_combustion_particles(".pc" + str(i + 1)))

    def destroy_particles(self):
        if len(self.post_combustion_particles) > 0:
            for i in range(len(self.engines_slots)):
                self.post_combustion_particles[i].destroy()
            self.post_combustion_particles = []
        if self.smoke is not None:
            self.smoke.destroy()
            self.smoke = None
        if self.explode is not None:
            self.explode.destroy()
            self.explode = None

        n = self.get_machinegun_count()
        for i in range(n):
            gmd = self.get_device("MachineGunDevice_%02d" % i)
            if gmd is not None:
                gmd.destroy_particles()

    def create_post_combustion_particles(self, engine_name):
        pc = ParticlesEngine(self.name + engine_name, self.scene, "bullet_impact", 15,
                             hg.Vec3(1, 1, 1), hg.Vec3(0.2, 0.2, 0.2), 15, 0)
        pc.delay_range = hg.Vec2(0.3, 0.4)
        pc.flow = 0
        pc.scale_range = hg.Vec2(1, 1)
        pc.start_speed_range = hg.Vec2(1, 1)
        pc.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0.9, 0.7, 0.5), hg.Color(0.9, 0.7, 0.1, 0.25),
                     hg.Color(0.9, 0.5, 0., 0.), hg.Color(0.85, 0.5, 0., 0.25), hg.Color(0.8, 0.4, 0., 0.15),
                     hg.Color(0.8, 0.1, 0.1, 0.05), hg.Color(0.5, 0., 0., 0.)]
        pc.set_rot_range(radians(1200), radians(2200), radians(1420), radians(1520), radians(1123), radians(5120))
        pc.gravity = hg.Vec3(0, 0, 0)
        pc.linear_damping = 1.0
        pc.loop = True
        return pc

    def start_explosion(self):

        if self.explode is not None:
            self.explode.flow = int(self.explode.num_particles * 5 / 4)
        if self.smoke is not None:
            self.smoke.reset_life_time(uniform(30, 60))

        self.set_thrust_level(0)
        n = self.get_machinegun_count()
        for i in range(n):
            mgd = self.get_device("MachineGunDevice_%02d" % i)
            if mgd is not None:
                mgd.stop_machine_gun()

        self.wreck = True

    def update_post_combustion_particles(self, dts, pos, rot_mat):
        for i, pcp in enumerate(self.post_combustion_particles):
            pos_prec = hg.GetT(self.engines_slots[i].GetTransform().GetWorld())
            pcp.update_kinetics(pos_prec + self.v_move * dts, hg.GetZ(rot_mat) * -1, self.v_move, hg.GetY(rot_mat), dts)

    def update_feedbacks(self, dts):
        if Destroyable_Machine.flag_activate_particles and Destroyable_Machine.flag_update_particles:

            mat, pos, rot, aX, aY, aZ = self.decompose_matrix()
            engines_pos = hg.Vec3(0, 0, 0)
            for i, pcp in enumerate(self.post_combustion_particles):
                engines_pos += hg.GetT(self.engines_slots[i].GetTransform().GetWorld())
            engines_pos /= len(self.post_combustion_particles)

            if self.smoke is not None:
                if (self.health_level < 1 or self.smoke.num_alive > 0) and not self.smoke.end:
                    self.smoke.update_kinetics(engines_pos, aZ * -1, self.v_move, aY, dts)  # AJOUTER UNE DUREE LIMITE AU FOURNEAU LORSQUE WRECK=TRUE !
            if self.explode is not None:
                if self.wreck and not self.explode.end:
                    self.explode.update_kinetics(pos, aZ * -1, self.v_move, aY, dts)

            self.update_post_combustion_particles(dts, pos, hg.GetRMatrix(mat))

    # ===================== Weapons ==============================

    def get_machines_guns_slots(self):
        return self.get_slots("machine_gun_slot")

    def get_machinegun_count(self):
        return len(self.machine_gun_slots)

    def get_num_bullets(self):
        n = self.get_machinegun_count()
        for i in range(n):
            gmd = self.get_device("MachineGunDevice_%02d" % i)
            if gmd is not None:
                n += gmd.get_num_bullets()
        return n

    # =========================== Missiles

    def get_missiles_slots(self):
        return self.get_slots("missile_slot")

    def get_num_missiles_slots(self):
        return len(self.missiles_slots)

    def rearm(self):
        self.set_health_level(1)
        n = self.get_machinegun_count()
        for i in range(n):
            gmd = self.get_device("MachineGunDevice_%02d" % i)
            if gmd is not None:
                gmd.reset()
        md = self.get_device("MissilesDevice")
        if md is not None:
            md.rearm()


    # ============================ Engines

    def get_engines_slots(self):
        return self.get_slots("engine_slot")

    def activate_easy_steering(self):
        if self.autopilot_activated or self.IA_activated:
            self.flag_easy_steering_mem = True
        else:
            self.flag_easy_steering = True

    def deactivate_easy_steering(self):
        if self.autopilot_activated or self.IA_activated:
            self.flag_easy_steering_mem = False
        else:
            self.flag_easy_steering = False

    def update_takoff_positionning(self, dts):
        self.v_move.x = self.v_move.y = self.v_move.z = 0
        self.t_going_to_takeoff_position += dts / 5
        if self.t_going_to_takeoff_position >= 1:
            self.flag_going_to_takeoff_position = False
        else:
            t = MathsSupp.smoothstep(0, 1, self.t_going_to_takeoff_position)
            pos = self.takeoff_position_start * (1 - t) + self.takeoff_position_dest * t
            self.parent_node.GetTransform().SetPos(pos)

    def set_landed(self):
        if not self.flag_landed:
            if self.ground_node_collision is not None:
                destroyable_machine = Destroyable_Machine.get_object_by_collision_node(self.ground_node_collision)
                if destroyable_machine is not None:
                    if destroyable_machine.nationality == self.nationality:
                        if destroyable_machine.type == Destroyable_Machine.TYPE_SHIP and not destroyable_machine.wreck:
                            mat = self.parent_node.GetTransform().GetWorld()
                            az = hg.GetZ(mat)
                            pos = hg.GetT(mat)
                            self.takeoff_position_start = pos
                            self.takeoff_position_dest = pos - az * 50
                            self.flag_going_to_takeoff_position = True
                            self.t_going_to_takeoff_position = 0
                            self.rearm()
            self.flag_landed = True

    def set_landing_targets(self, targets):
        self.landing_targets = targets

    # ============================= Physics ====================================

    def compute_z_drag(self):
        if "Gear" in self.devices and self.devices["Gear"] is not None:
            gear = self.devices["Gear"]
            gear_lvl = gear.gear_level
        else:
            gear_lvl = 0
        return self.drag_coeff.z + self.brake_drag * self.brake_level + self.flaps_level * self.flaps_drag + self.gear_drag * gear_lvl

    def stabilize(self, p, y, r):
        if p: self.set_pitch_level(0)
        if y: self.set_yaw_level(0)
        if r: self.set_roll_level(0)

    def update_inertial_value(self, v0, vd, vi, dts):
        vt = vd - v0
        if vt < 0:
            v = v0 - vi * dts
            if v < vd: v = vd
        elif vt > 0:
            v = v0 + vi * dts
            if v > vd: v = vd
        else:
            v = vd
        return v

    def update_collisions(self, matrix, dts):

        mat, pos, rot, aX, aY, aZ = self.decompose_matrix(matrix)

        collisions_raycasts = [
            {"name": "down", "position": self.bound_down, "direction": hg.Vec3(0, -5, 0)}
        ]

        ray_hits, self.terrain_altitude, self.terrain_normale = update_collisions(mat, self, collisions_raycasts)

        alt = self.terrain_altitude
        if "Gear" in self.devices and self.devices["Gear"] is not None and self.devices["Gear"].activated:
            bottom_alt = self.devices["Gear"].gear_height
        else:
            bottom_alt = self.bottom_height
        self.ground_node_collision = None

        for collision in ray_hits:
            if collision["name"] == "down":
                if len(collision["hits"]) > 0:
                    hit = collision["hits"][0]
                    machine = self.get_machine_by_node(hit.node)
                    if machine is not None and machine.type != Destroyable_Machine.TYPE_SHIP and machine.type != Destroyable_Machine.TYPE_GROUND:
                        self.hit(1)
                    else:
                        self.ground_node_collision = hit.node
                        alt = hit.P.y + bottom_alt

        self.flag_landing = False

        if self.flag_crashed:
            pos.y = alt
            self.v_move *= pow(0.9, 60 * dts)

        else:
            if pos.y < alt:
                flag_crash = True
                if "Gear" in self.devices and self.devices["Gear"] is not None:
                    gear = self.devices["Gear"]
                    linear_speed = self.get_linear_speed()
                    if gear.activated and degrees(abs(asin(aZ.y))) < 15 and degrees(abs(asin(aX.y))) < 10 and linear_speed * 3.6 < self.landing_max_speed:

                        pos.y += (alt - pos.y) * 0.1 * 60 * dts
                        if self.v_move.y < 0: self.v_move.y *= pow(0.8, 60 * dts)
                        # b = min(1, self.brake_level + (1 - health_wreck_factor))
                        b = self.brake_level
                        self.v_move *= ((b * pow(0.98, 60 * dts)) + (1 - b))
                        # r=self.parent_node.GetTransform().GetRot()
                        f = ((b * pow(0.95, 60 * dts)) + (1 - b))
                        rot.x *= f
                        rot.z *= f
                        # self.parent_node.GetTransform().SetRot(rot)
                        self.flag_landing = True
                        flag_crash = False

                if flag_crash:
                    pos.y = alt
                    self.crash()

        return hg.TransformationMat4(pos, rot)

    def crash(self):
        self.hit(1)
        self.flag_crashed = True
        self.set_thrust_level(0)
        ia_ctrl = self.get_device("IAControlDevice")
        if ia_ctrl is not None:
            ia_ctrl.deactivate()
        else:
            ap_ctrl = self.get_device("AutopilotControlDevice")
            if ap_ctrl is not None:
                ap_ctrl.deactivate()

    def update_angular_levels(self, dts):
        self.angular_levels.x = self.update_inertial_value(self.angular_levels.x, self.angular_levels_dest.x,
                                                           self.angular_levels_inertias.x, dts)
        self.angular_levels.y = self.update_inertial_value(self.angular_levels.y, self.angular_levels_dest.y,
                                                           self.angular_levels_inertias.y, dts)
        self.angular_levels.z = self.update_inertial_value(self.angular_levels.z, self.angular_levels_dest.z,
                                                           self.angular_levels_inertias.z, dts)

    # ==================================================================

    def get_physics_parameters(self):
        # ============================ Compute Thrust impulse
        tf = self.thrust_force
        if self.post_combustion and self.thrust_level == 1:
            tf += self.post_combustion_force
        # ================================ Compute Z drag impulse
        dc = hg.Vec3(self.drag_coeff)
        dc.z = self.compute_z_drag()

        return {"v_move": self.v_move,
                "thrust_level": self.thrust_level,
                "thrust_force": tf,
                "lift_force": self.wings_lift + self.flaps_level * self.flaps_lift,
                "drag_coefficients": dc,
                "health_wreck_factor": pow(self.health_level, 0.2),
                "angular_levels": self.angular_levels,
                "angular_frictions": self.angular_frictions,
                "speed_ceiling": self.speed_ceiling,
                "flag_easy_steering": self.flag_easy_steering
                }

    def update_kinetics(self, dts):

        # Custom physics (but keep inner collisions system)
        if self.flag_custom_physics_mode:
            Destroyable_Machine.update_kinetics(self, dts)

        # Inner physics
        else:

            if self.activated:
                self.update_devices(dts) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                # # ========================= Flight physics Repositionning after landing :

                if self.flag_going_to_takeoff_position:
                    self.update_takoff_positionning(dts)

                # ========================= Flight physics
                else:

                    # ==================== Aircraft Inertias for animated parts
                    self.update_thrust_level(dts)  # Applies disfunctions
                    self.update_brake_level(dts)  # Brake level inertia
                    self.update_flaps_level(dts)  # Flaps level inertia
                    self.update_angular_levels(dts)  # flaps inertia

                    self.update_mobile_parts(dts)  # Animate mobile parts

                    # # ========================= Update physics

                    physics_parameters = self.get_physics_parameters()

                    mat, physics_parameters = update_physics(self.parent_node.GetTransform().GetWorld(), self, physics_parameters, dts)

                    # ======================== Update aircraft vars:

                    self.pitch_attitude = physics_parameters["pitch_attitude"]
                    self.heading = physics_parameters["heading"]
                    self.roll_attitude = physics_parameters["roll_attitude"]
                    self.v_move = physics_parameters["v_move"]

                    # ========================== Update collisions

                    mat = self.update_collisions(mat, dts)

                    # Landed state:
                    ia_ctrl = self.get_device("IAControlDevice")
                    if not self.flag_crashed and not ia_ctrl.is_activated():
                        hs, vs = self.get_world_speed()
                        if abs(vs) > 1:
                            self.flag_landed = False
                        if hs < 1 and abs(vs) < 1:
                            self.set_landed()

                    # ======== Update matrix==========================================================

                    mat, pos, rot, aX, aY, aZ = self.decompose_matrix(mat)

                    self.parent_node.GetTransform().SetPos(pos)
                    self.parent_node.GetTransform().SetRot(rot)

                    # ======== Update Acceleration ==========================================================

                    self.rec_linear_speed()
                    self.update_linear_acceleration()


                # ====================== Update Feed backs:

                self.update_feedbacks(dts)

    def gui(self):
        if hg.ImGuiBegin("Aircraft"):

            hg.ImGuiSetWindowPos("Aircraft", hg.Vec2(1300, 20), hg.ImGuiCond_Once)
            hg.ImGuiSetWindowSize("Aircraft", hg.Vec2(600, 450), hg.ImGuiCond_Once)
            hg.ImGuiText("Name:" + self.name)
            hg.ImGuiText("Altitude: " + str(int(self.get_altitude())))
            hg.ImGuiText("Cap:" + str(int(self.heading)))
            hg.ImGuiText("Health:" + str(int(self.health_level * 100)) + "%")
            hg.ImGuiText("Linear speed:" + str(int(self.get_linear_speed() * 3.6)))
            hs, vs = self.get_world_speed()
            hg.ImGuiText("Horizontal speed:" + str(int(hs * 3.6)))
            hg.ImGuiText("Vertical speed:" + str(int(vs * 3.6)))

            d, self.flag_display_linear_speed = hg.ImGuiCheckbox("Display linear speed", self.flag_display_linear_speed)
            d, self.flag_display_vertical_speed = hg.ImGuiCheckbox("Display vertical speed", self.flag_display_vertical_speed)
            d, self.flag_display_horizontal_speed = hg.ImGuiCheckbox("Display horizontal speed", self.flag_display_horizontal_speed)

            if hg.ImGuiButton("RESET"):
                self.reset()

            ia_ctrl = self.get_device("IAControlDevice")
            if ia_ctrl is not None:
                d, f = hg.ImGuiCheckbox("IA activated", ia_ctrl.is_activated())
                if d:
                    if f:
                        ia_ctrl.activate()
                    else:
                        ia_ctrl.deactivate()

                if ia_ctrl.is_activated():
                    hg.ImGuiText("IA command: " + ia_ctrl.IA_commands_labels[ia_ctrl.IA_command])


            hg.ImGuiSeparator()
            ap_ctrl = self.get_device("AutopilotControlDevice")
            if ap_ctrl is not None:
                if ap_ctrl.is_user_control_active():
                    d, f = hg.ImGuiCheckbox("Autopilot activated", ap_ctrl.is_activated())
                    if d:
                        if f:
                            ap_ctrl.activate()
                        else:
                            ap_ctrl.deactivate()

                    d, f = hg.ImGuiSliderFloat("Autopilot heading", ap_ctrl.autopilot_heading, 0, 360)
                    if d:
                        ap_ctrl.set_autopilot_heading(f)

                    d, f = hg.ImGuiSliderFloat("Autopilot altitude (m)", ap_ctrl.autopilot_altitude, ap_ctrl.altitude_range[0], ap_ctrl.altitude_range[1])
                    if d:
                        ap_ctrl.set_autopilot_altitude(f)

                    d, f = hg.ImGuiSliderFloat("Autopilot speed (km/h)", ap_ctrl.autopilot_speed * 3.6, ap_ctrl.speed_range[0], ap_ctrl.speed_range[1])
                    if d:
                        ap_ctrl.set_autopilot_speed(f / 3.6)
                else:
                    hg.ImGuiText("Autopilot heading: %.2f" % (ap_ctrl.autopilot_heading))
                    hg.ImGuiText("Autopilot altitude (m): %.2f" % (ap_ctrl.autopilot_altitude))
                    hg.ImGuiText("Autopilot speed (km/h): %.2f" % (ap_ctrl.autopilot_speed * 3.6))

            td = self.get_device("TargettingDevice")
            if td is not None:
                targets_list = hg.StringList()
                targets_list.push_back("- None -")

                for target in td.targets:
                    nm = target.name
                    if target.wreck:
                        nm += " - WRECK!"
                    if not target.activated:
                        nm += " - INACTIVE!"
                    targets_list.push_back(nm)

                f, d = hg.ImGuiListBox("Targets", td.target_id, targets_list, 20)
                if f:
                    td.set_target_id(d)

        hg.ImGuiEnd()


# ========================================================================================================
#                   Sounds handlers
# ========================================================================================================


class MissileSFX:

    def __init__(self, missile: Missile):
        self.missile = missile

        self.explosion_source = None
        self.explosion_state = create_spatialized_sound_state(hg.SR_Once)
        self.explosion_ref = hg.LoadWAVSoundAsset("sfx/missile_explosion.wav")

        self.turbine_ref = hg.LoadWAVSoundAsset("sfx/missile_engine.wav")
        self.turbine_source = None
        self.turbine_state = create_spatialized_sound_state(hg.SR_Loop)

        self.start = False
        self.exploded = False

    def reset(self):
        self.exploded = False

    def start_engine(self, main):
        self.turbine_state.volume = 0
        # self.turbine_state.pitch = 1
        self.turbine_source = hg.PlaySpatialized(self.turbine_ref, self.turbine_state)
        self.start = True

    def stop_engine(self, main):
        self.turbine_state.volume = 0
        # self.turbine_state.pitch = 1
        hg.StopSource(self.turbine_source)
        self.turbine_source = None

    def update_sfx(self, main, dts):
        if self.missile.activated:
            self.missile.calculate_view_matrix(main.scene.GetCurrentCamera())
            self.missile.update_view_v_move(dts)

            level = MathsSupp.get_sound_distance_level(hg.GetT(self.missile.mat_view)) * main.master_sfx_volume

            if not self.start:
                self.start_engine(main)

            if self.missile.wreck and not self.exploded:
                self.explosion_state.volume = level
                self.stop_engine(main)
                self.explosion_source = hg.PlaySpatialized(self.explosion_ref, self.explosion_state)
                self.exploded = True

            if not self.exploded:
                self.turbine_state.volume = 0.5 * level
                # self.turbine_state.pitch = self.turbine_pitch_levels.x + self.aircraft.thrust_level * (self.turbine_pitch_levels.y - self.turbine_pitch_levels.x)

                hg.SetSourceTransform(self.turbine_source, self.missile.mat_view, self.missile.view_v_move)
                hg.SetSourceVolume(self.turbine_source, self.turbine_state.volume)

            else:
                hg.SetSourceTransform(self.explosion_source, self.missile.mat_view, self.missile.view_v_move)
                hg.SetSourceVolume(self.explosion_source, min(1, level * 2))


class AircraftSFX:

    def __init__(self, aircraft: Aircraft):
        self.aircraft = aircraft

        self.turbine_pitch_levels = hg.Vec2(1, 2)

        self.turbine_ref = hg.LoadWAVSoundAsset("sfx/turbine.wav")
        self.air_ref = hg.LoadWAVSoundAsset("sfx/air.wav")
        self.pc_ref = hg.LoadWAVSoundAsset("sfx/post_combustion.wav")
        self.wind_ref = hg.LoadWAVSoundAsset("sfx/wind.wav")
        self.explosion_ref = hg.LoadWAVSoundAsset("sfx/explosion.wav")
        self.machine_gun_ref = hg.LoadWAVSoundAsset("sfx/machine_gun.wav")
        self.burning_ref = hg.LoadWAVSoundAsset("sfx/burning.wav")

        self.turbine_state = create_spatialized_sound_state(hg.SR_Loop)
        self.air_state = create_spatialized_sound_state(hg.SR_Loop)
        self.pc_state = create_spatialized_sound_state(hg.SR_Loop)
        self.wind_state = create_spatialized_sound_state(hg.SR_Loop)
        self.explosion_state = create_spatialized_sound_state(hg.SR_Once)
        self.machine_gun_state = create_spatialized_sound_state(hg.SR_Once)
        self.burning_state = create_spatialized_sound_state(hg.SR_Loop)

        self.start = False

        self.pc_cptr = 0
        self.pc_start_delay = 0.25
        self.pc_stop_delay = 0.5

        self.turbine_source = None
        self.wind_source = None
        self.air_source = None
        self.pc_source = None
        self.explosion_source = None
        self.machine_gun_source = None
        self.burning_source = None

        self.pc_started = False
        self.pc_stopped = False

        self.exploded = False

    def reset(self):
        self.exploded = False

    def set_air_pitch(self, value):
        self.air_state.pitch = value

    def set_pc_pitch(self, value):
        self.pc_state.pitch = value

    def set_turbine_pitch_levels(self, values: hg.Vec2):
        self.turbine_pitch_levels = values

    def start_engine(self, main):
        self.turbine_state.volume = 0
        # self.turbine_state.pitch = 1
        self.air_state.volume = 0
        self.pc_state.volume = 0
        self.air_source = hg.PlaySpatialized(self.air_ref, self.air_state)
        self.turbine_source = hg.PlaySpatialized(self.turbine_ref, self.turbine_state)
        self.pc_source = hg.PlaySpatialized(self.pc_ref, self.pc_state)
        self.start = True
        self.pc_started = False
        self.pc_stopped = True

    def stop_engine(self, main):
        self.turbine_state.volume = 0
        # self.turbine_state.pitch = 1
        self.air_state.volume = 0
        self.pc_state.volume = 0
        if self.turbine_source is not None:
            hg.StopSource(self.turbine_source)
        if self.air_source is not None:
            hg.StopSource(self.air_source)
        if self.pc_source is not None:
            hg.StopSource(self.pc_source)

        self.start = False
        self.pc_started = False
        self.pc_stopped = True

        # self.wind_state.volume = 0

    def update_sfx(self, main, dts):

        self.aircraft.calculate_view_matrix(main.scene.GetCurrentCamera())
        self.aircraft.update_view_v_move(dts)

        level = MathsSupp.get_sound_distance_level(hg.GetT(self.aircraft.mat_view)) * main.master_sfx_volume

        if self.aircraft.thrust_level > 0 and not self.start:
            self.start_engine(main)

        if self.aircraft.smoke is not None:
            if self.aircraft.health_level < 1 and not self.aircraft.smoke.end:
                self.burning_state.volume = level * (1 - self.aircraft.health_level) * self.aircraft.smoke.life_f
                self.burning_state.mtx = self.aircraft.mat_view
                self.burning_state.vel = self.aircraft.view_v_move
                if self.burning_source is None:
                    self.burning_source = hg.PlaySpatialized(self.burning_ref, self.burning_state)
                hg.SetSourceTransform(self.burning_source, self.burning_state.mtx, self.burning_state.vel)
                hg.SetSourceVolume(self.burning_source, self.burning_state.volume)

            elif self.burning_source is not None:
                hg.StopSource(self.burning_source)
                self.burning_source = None

        if self.aircraft.wreck and not self.exploded:
            self.explosion_state.volume = level
            self.stop_engine(main)
            self.explosion_source = hg.PlaySpatialized(self.explosion_ref, self.explosion_state)
            self.exploded = True

        if self.start:
            if self.aircraft.thrust_level <= 0:
                self.stop_engine(main)

            else:
                self.turbine_state.volume = 0.5 * level
                # self.turbine_state.pitch = self.turbine_pitch_levels.x + self.aircraft.thrust_level * (self.turbine_pitch_levels.y - self.turbine_pitch_levels.x)
                self.air_state.volume = (0.1 + self.aircraft.thrust_level * 0.9) * level

                if self.aircraft.post_combustion:
                    self.pc_state.volume = level
                    if not self.pc_started:
                        self.pc_stopped = False
                        self.pc_state.volume *= self.pc_cptr / self.pc_start_delay
                        self.pc_cptr += dts
                        if self.pc_cptr >= self.pc_start_delay:
                            self.pc_started = True
                            self.pc_cptr = 0

                else:
                    if not self.pc_stopped:
                        self.pc_started = False
                        self.pc_state.volume = (1 - self.pc_cptr / self.pc_stop_delay) * level
                        self.pc_cptr += dts
                        if self.pc_cptr >= self.pc_stop_delay:
                            self.pc_stopped = True
                            self.pc_cptr = 0

                hg.SetSourceTransform(self.turbine_source, self.aircraft.mat_view, self.aircraft.view_v_move)
                hg.SetSourceVolume(self.turbine_source, self.turbine_state.volume)
                hg.SetSourceTransform(self.air_source, self.aircraft.mat_view, self.aircraft.view_v_move)
                hg.SetSourceVolume(self.air_source, self.air_state.volume)
                hg.SetSourceTransform(self.pc_source, self.aircraft.mat_view, self.aircraft.view_v_move)
                hg.SetSourceVolume(self.pc_source, self.pc_state.volume)

        if self.explosion_source is not None:
            hg.SetSourceTransform(self.explosion_source, self.aircraft.mat_view, self.aircraft.view_v_move)
            hg.SetSourceVolume(self.explosion_source, min(1, level * 2))

        f = min(1, self.aircraft.get_linear_speed() * 3.6 / 1000)
        self.wind_state.volume = f * level
        self.wind_state.mtx = self.aircraft.mat_view
        self.wind_state.vel = self.aircraft.view_v_move
        if self.wind_source is None:
            self.wind_source = hg.PlaySpatialized(self.wind_ref, self.wind_state)
        hg.SetSourceTransform(self.wind_source, self.wind_state.mtx, self.wind_state.vel)
        hg.SetSourceVolume(self.wind_source, self.wind_state.volume)

        # Machine gun
        n = self.aircraft.get_machinegun_count()
        if n > 0:
            num_new = 0
            for i in range(n):
                gmd = self.aircraft.get_device("MachineGunDevice_%02d" % i)
                if gmd is not None:
                    num_new += gmd.get_new_bullets_count()
            if num_new > 0:
                self.machine_gun_state.volume = level * 0.5
                self.machine_gun_state.mtx = self.aircraft.mat_view
                self.machine_gun_state.vel = self.aircraft.view_v_move
                self.machine_gun_source = hg.PlaySpatialized(self.machine_gun_ref, self.machine_gun_state)


# =====================================================================================================
#                                   Aircraft-carrier
# =====================================================================================================


class Carrier(Destroyable_Machine):
    instance_scene_name = "machines/aircraft_carrier_blend/aircraft_carrier_blend.scn"

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Destroyable_Machine.__init__(self, name, "Basic_Carrier", scene, scene_physics, pipeline_ressource, Carrier.instance_scene_name, Destroyable_Machine.TYPE_SHIP, nationality)
        self.setup_collisions()
        self.activated = True
        sv = self.parent_node.GetInstanceSceneView()
        self.radar = sv.GetNode(scene, "aircraft_carrier_radar")
        self.fps_start_point = sv.GetNode(scene, "fps_start_point")
        self.aircraft_start_points = []
        self.landing_points = []
        self.find_nodes("carrier_aircraft_start_point.", self.aircraft_start_points, 1)
        self.find_nodes("landing_point.", self.landing_points, 1)
        self.landing_targets = []
        for landing_point in self.landing_points:
            self.landing_targets.append(LandingTarget(landing_point))

    def find_nodes(self, name, tab, first=1):
        i = first
        sv = self.parent_node.GetInstanceSceneView()
        n = sv.GetNodes(self.scene).size()
        if n == 0:
            raise OSError("ERROR - Empty Instance '" + self.name + "'- Unloaded scene ?")
        while True:
            nm = name + '{:03d}'.format(i)
            nd = sv.GetNode(self.scene, nm)
            node_name = nd.GetName()
            if node_name != nm:
                break
            else:
                tab.append(nd)
                i += 1

    def destroy(self):
        if not self.flag_destroyed:
            self.destroy_nodes()
            self.flag_destroyed = True

    def hit(self, value):
        pass

    def update_kinetics(self, dts):
        rot = self.radar.GetTransform().GetRot()
        rot.y += radians(45 * dts)
        self.radar.GetTransform().SetRot(rot)

    def get_aircraft_start_point(self, point_id):
        mat = self.aircraft_start_points[point_id].GetTransform().GetWorld()
        return hg.GetT(mat), hg.GetR(mat)

# =====================================================================================================
#                                   LandVehicle
# =====================================================================================================

class LandVehicle(Destroyable_Machine):

    def __init__(self, name, model_name, scene, scene_physics, pipeline_ressource, instance_scene_name, nationality, start_position=None, start_rotation=None):
        Destroyable_Machine.__init__(self, name, model_name, scene, scene_physics, pipeline_ressource, instance_scene_name, Destroyable_Machine.TYPE_LANDVEHICLE, nationality, start_position, start_rotation)

        self.thrust_level = 0
        self.brake_level = 0

        self.setup_bounds_positions()

    def destroy(self):
        self.destroy_nodes()
        self.flag_destroyed = True

    def get_thrust_level(self):
        return self.thrust_level

    def get_brake_level(self):
        return self.brake_level

    def update_kinetics(self, dts):
        Destroyable_Machine.update_kinetics(self, dts)
main = None
dogfight_network_port = 50888
flag_server_running = False
flag_server_connected = False
server_log = ""
listen_socket_thread = None
flag_print_log = True

commands_functions = []


def get_network():
	return HOST, dogfight_network_port


def init_server(main_):
	global server_log, commands_functions, main
	main = main_

	commands_functions = {

		# Globals
		"DISABLE_LOG": disable_log,
		"ENABLE_LOG": enable_log,
		"GET_RUNNING": get_running,
		"SET_RENDERLESS_MODE": set_renderless_mode,
		"SET_DISPLAY_RADAR_IN_RENDERLESS_MODE": set_display_radar_in_renderless_mode,
		"SET_TIMESTEP": set_timestep,
		"GET_TIMESTEP": get_timestep,
		"SET_CLIENT_UPDATE_MODE": set_client_update_mode,
		"UPDATE_SCENE": update_scene,
		"DISPLAY_VECTOR": display_vector,
		"DISPLAY_2DTEXT": display_2DText,

		# Machines
		"GET_MACHINE_MISSILES_LIST": get_machine_missiles_list,
		"GET_TARGETS_LIST": get_targets_list,
		"GET_HEALTH": get_health,
		"SET_HEALTH": set_health,
		"ACTIVATE_AUTOPILOT": activate_autopilot,
		"DEACTIVATE_AUTOPILOT": deactivate_autopilot,
		"ACTIVATE_IA": activate_IA,
		"DEACTIVATE_IA": deactivate_IA,
		"GET_MACHINE_GUN_STATE": get_machine_gun_state,
		"ACTIVATE_MACHINE_GUN": activate_machine_gun,
		"DEACTIVATE_MACHINE_GUN": deactivate_machine_gun,
		"GET_MISSILESDEVICE_SLOTS_STATE": get_missiles_device_slots_state,
		"FIRE_MISSILE": fire_missile,
		"REARM_MACHINE": rearm_machine,
		"GET_TARGET_IDX": get_target_idx,
		"SET_TARGET_ID": set_target_id,
		"RESET_MACHINE_MATRIX": reset_machine_matrix,
		"RESET_MACHINE": reset_machine,
		"SET_MACHINE_CUSTOM_PHYSICS_MODE": set_machine_custom_physics_mode,
		"GET_MACHINE_CUSTOM_PHYSICS_MODE": get_machine_custom_physics_mode,
		"UPDATE_MACHINE_KINETICS": update_machine_kinetics,
		"GET_MOBILE_PARTS_LIST": get_mobile_parts_list,
		"IS_AUTOPILOT_ACTIVATED": is_autopilot_activated,
		"ACTIVATE_AUTOPILOT": activate_autopilot,
		"DEACTIVATE_AUTOPILOT": deactivate_autopilot,
		"IS_IA_ACTIVATED": is_ia_activated,
		"ACTIVATE_IA": activate_IA,
		"DEACTIVATE_IA": deactivate_IA,
		"IS_USER_CONTROL_ACTIVATED": is_user_control_activated,
		"ACTIVATE_USER_CONTROL": activate_user_control,
		"DEACTIVATE_USER_CONTROL": deactivate_user_control,
		"COMPUTE_NEXT_TIMESTEP_PHYSICS": compute_next_timestep_physics,

		# Aircrafts
		"GET_PLANESLIST": get_planes_list,
		"GET_PLANE_STATE": get_plane_state,
		"SET_PLANE_THRUST": set_plane_thrust,
		"GET_PLANE_THRUST": get_plane_thrust,
		"ACTIVATE_PC": activate_pc,
		"DEACTIVATE_PC": deactivate_pc,
		"SET_PLANE_BRAKE": set_plane_brake,
		"SET_PLANE_FLAPS": set_plane_flaps,
		"SET_PLANE_PITCH": set_plane_pitch,
		"SET_PLANE_ROLL": set_plane_roll,
		"SET_PLANE_YAW": set_plane_yaw,
		"STABILIZE_PLANE": stabilize_plane,
		"DEPLOY_GEAR": deploy_gear,
		"RETRACT_GEAR": retract_gear,
		"SET_PLANE_AUTOPILOT_SPEED": set_plane_autopilot_speed,
		"SET_PLANE_AUTOPILOT_HEADING": set_plane_autopilot_heading,
		"SET_PLANE_AUTOPILOT_ALTITUDE": set_plane_autopilot_altitude,
		"ACTIVATE_PLANE_EASY_STEERING": activate_plane_easy_steering,
		"DEACTIVATE_PLANE_EASY_STEERING": deactivate_plane_easy_steering,
		"SET_PLANE_LINEAR_SPEED": set_plane_linear_speed,
		"RESET_GEAR": reset_gear,
		"RECORD_PLANE_START_STATE": record_plane_start_state,

		# Missiles
		"GET_MISSILESLIST": get_missiles_list,
		"GET_MISSILE_STATE": get_missile_state,
		"SET_MISSILE_LIFE_DELAY": set_missile_life_delay,
		"GET_MISSILE_TARGETS_LIST": get_missile_targets_list,
		"SET_MISSILE_TARGET": set_missile_target,
		"SET_MISSILE_THRUST_FORCE": set_missile_thrust_force,
		"SET_MISSILE_ANGULAR_FRICTIONS": set_missile_angular_frictions,
		"SET_MISSILE_DRAG_COEFFICIENTS": set_missile_drag_coefficients,

		# Missile launchers
		"GET_MISSILE_LAUNCHERS_LIST": get_missile_launchers_list,
		"GET_MISSILE_LAUNCHER_STATE": get_missile_launcher_state
	}
	server_log = ""
	msg = "Hostname: %s, IP: %s, port: %d" % (hostname, HOST, dogfight_network_port)
	server_log = msg
	print(msg)


def start_server():
	global flag_server_running, listen_socket_thread
	main.flag_client_connected = False
	flag_server_running = True
	listen_socket_tread = threading.Thread(target=server_update)
	listen_socket_tread.start()


def stop_server():
	global flag_server_running, flag_server_connected, server_log
	flag_server_running = False
	flag_server_connected = False
	main.flag_client_connected = False
	msg = "Exit from server"
	server_log += msg
	print(msg)


def server_update():
	global flag_server_running, server_log, flag_server_connected

	while flag_server_running:
		try:
			listener_socket(dogfight_network_port)
			print(logger)
			server_log += logger
			flag_server_connected = True
			main.flag_client_connected = True
			while flag_server_connected:
				answ = get_answer()
				answ.decode()
				command = json.loads(answ)
				if command == "":
					server_log += "Disconnected"
					flag_server_connected = False
					main.flag_client_connected = False
				else:
					msg = "command:" + command["command"]
					if flag_print_log:
						print(msg)
						server_log += msg
					commands_functions[command["command"]](command["args"])

		except:
			print("network_server.py - server_update ERROR")
			flag_server_connected = False
			main.flag_client_connected = False
			main.flag_client_update_mode = False
			main.set_renderless_mode(False)
			msg = "Socket closed"
			server_log += msg
			print(msg)


# Globals

def disable_log(args):
	global flag_print_log
	flag_print_log = False


def enable_log(args):
	global flag_print_log
	flag_print_log = True


def update_scene(args):
	if main.flag_client_update_mode:
		if main.flag_renderless:
			main.update() # No display, but fast simulation
		else:
			main.flag_client_ask_update_scene = True # display simulation at 60 fps
	elif flag_print_log:
		print("Update_scene ERROR - Client update mode is FALSE")


def display_vector(args):
	if main.flag_client_update_mode:
		position = hg.Vec3(args["position"][0], args["position"][1], args["position"][2])
		direction = hg.Vec3(args["direction"][0], args["direction"][1], args["direction"][2])
		label_offset2D = hg.Vec2(args["label_offset2D"][0], args["label_offset2D"][1])
		color = hg.Color(args["color"][0],args["color"][1], args["color"][2], args["color"][3])
		Overlays.display_named_vector(position, direction, args["label"], label_offset2D, color, args["label_size"])
	elif flag_print_log:
		print("Display vector ERROR - Client Update Mode must be TRUE")


def display_2DText(args):
	if main.flag_client_update_mode:
		position = hg.Vec2(args["position"][0], args["position"][1])
		color = hg.Color(args["color"][0],args["color"][1], args["color"][2], args["color"][3])
		Overlays.add_text2D(args["text"], position, args["size"], color)
	elif flag_print_log:
		print("Display 2d Text ERROR - Client Update Mode must be TRUE")


def set_timestep(args):
	main.timestep = args["timestep"]


def get_timestep(args):
	ts = {"timestep": main.timestep}
	send_message(str.encode(json.dumps(ts)))


def get_running(args):
	state = {"running": main.flag_running}
	send_message(str.encode(json.dumps(state)))


def set_renderless_mode(args):
	main.set_renderless_mode(args["flag"])


def set_display_radar_in_renderless_mode(args):
	main.flag_display_radar_in_renderless = args["flag"]


def set_client_update_mode(args):
	main.flag_client_update_mode = args["flag"]


# Machines


def set_machine_custom_physics_mode(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	machine.set_custom_physics_mode(args["flag"])


def get_machine_custom_physics_mode(args):
	machine = main.destroyables_items[args["machine_id"]]
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"custom_physics_mode": machine.flag_custom_physics_mode
	}
	if flag_print_log:
		print(args["machine_id"])
		print(str(state))
	send_message(str.encode(json.dumps(state)))


def update_machine_kinetics(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	print("c0")
	mat = hg.TranslationMat4(hg.Vec3(0, 0, 0))
	for n in args["matrix"]:
		if math.isnan(n):
			args["matrix"] = [1, 0, 0,
							0, 1, 0,
							0, 0, 1,
							0, 200, 0]
			break
	for n in args["v_move"]:
		if math.isnan(n):
			args["v_move"] = [0, 0, 0]
			break
	hg.SetRow(mat, 0, hg.Vec4(args["matrix"][0], args["matrix"][3], args["matrix"][6], args["matrix"][9]))
	hg.SetRow(mat, 1, hg.Vec4(args["matrix"][1], args["matrix"][4], args["matrix"][7], args["matrix"][10]))
	hg.SetRow(mat, 2, hg.Vec4(args["matrix"][2], args["matrix"][5], args["matrix"][8], args["matrix"][11]))
	machine.set_custom_kinetics(mat, hg.Vec3(args["v_move"][0], args["v_move"][1], args["v_move"][2]))


def reset_machine(args):
	machine = main.destroyables_items[args["machine_id"]]
	machine.reset()


def reset_machine_matrix(args):
	machine = main.destroyables_items[args["machine_id"]]
	pos = hg.Vec3(args["position"][0], args["position"][1], args["position"][2])
	rot = hg.Vec3(args["rotation"][0], args["rotation"][1], args["rotation"][2])
	machine.reset_matrix(pos, rot)
	if machine.type == Destroyable_Machine.TYPE_AIRCRAFT:
		machine.flag_landed = False


def get_machine_missiles_list(args):
	machine = main.destroyables_items[args["machine_id"]]
	missiles = []
	md = machine.get_device("MissilesDevice")
	if md is not None:
		for missile in md.missiles:
			if missile is None:
				missiles.append("")
			else:
				missiles.append(missile.name)
	else:
		print("ERROR - Machine '" + args["machine_id"] + "' has no MissilesDevice !")
	send_message(str.encode(json.dumps(missiles)))


def get_targets_list(args):
	machine = main.destroyables_items[args["machine_id"]]
	td = machine.get_device("TargettingDevice")
	if td is not None:
		tlist = [{"target_id": "-None-"}]  # Target id 0 = no target
		targets = td.get_targets()
		for target in targets:
			tlist.append({"target_id": target.name, "wreck": target.wreck, "active": target.activated})
		if flag_print_log:
			print(args["machine_id"])
			print(str(tlist))
	else:
		tlist = []
		print("ERROR - Machine '" + args["machine_id"] + "' has no TargettingDevice !")
	send_message(str.encode(json.dumps(tlist)))


def get_mobile_parts_list(args):
	machine = main.destroyables_items[args["machine_id"]]
	parts = machine.get_mobile_parts()
	parts_id = []
	for part_id in parts:
		parts_id.append(part_id)
	send_message(str.encode(json.dumps(parts_id)))


def get_machine_gun_state(args):
	machine = main.destroyables_items[args["machine_id"]]
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"MachineGunDevices": {}
	}
	n = machine.get_machinegun_count()
	if n > 0:
		for i in range(n):
			gm_name = "MachineGunDevice_%02d" % i
			gmd = machine.get_device(gm_name)
			if gmd is not None:
				gm_state = {
					"machine_gun_activated": gmd.is_gun_activated(),
					"bullets_count": gmd.get_num_bullets()
				}
				state["MachineGunDevices"][gm_name] = gm_state
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
		send_message(str.encode(json.dumps(state)))
	else:
		print("ERROR - Machine '" + args["machine_id"] + "' has no MachineGunDevice !")


def get_missiles_device_slots_state(args):
	machine = main.destroyables_items[args["machine_id"]]
	md = machine.get_device("MissilesDevice")
	if md is not None:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"missiles_slots": md.get_missiles_state()
		}
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
		send_message(str.encode(json.dumps(state)))
	else:
		print("ERROR - Machine '" + args["machine_id"] + "' has no MissilesDevice !")


def fire_missile(args):
	if flag_print_log:
		print(args["machine_id"] + " " + str(args["slot_id"]))
	machine = main.destroyables_items[args["machine_id"]]
	md = machine.get_device("MissilesDevice")
	if md is not None:
		md.fire_missile(int(args["slot_id"]))
	else:
		print("ERROR - Machine '" + args["machine_id"] + "' has no MissilesDevice !")


def rearm_machine(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	machine.rearm()


def activate_machine_gun(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	n = machine.get_machinegun_count()
	for i in range(n):
		mgd = machine.get_device("MachineGunDevice_%02d" % i)
		if mgd is not None and not mgd.is_gun_activated():
			mgd.fire_machine_gun()


def deactivate_machine_gun(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	n = machine.get_machinegun_count()
	for i in range(n):
		mgd = machine.get_device("MachineGunDevice_%02d" % i)
		if mgd is not None and mgd.is_gun_activated():
			mgd.stop_machine_gun()


def get_health(args):
	machine = main.destroyables_items[args["machine_id"]]
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"health_level": machine.get_health_level()
	}
	if flag_print_log:
		print(args["machine_id"])
		print(str(state))
	send_message(str.encode(json.dumps(state)))


def set_health(args):
	if flag_print_log:
		print(args["machine_id"] + " " + str(args["health_level"]))
	machine = main.destroyables_items[args["machine_id"]]
	machine.set_health_level(args["health_level"])


def get_target_idx(args):
	machine = main.destroyables_items[args["machine_id"]]
	td = machine.get_device("TargettingDevice")
	if td is not None:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"target_idx": td.get_target_id()
		}
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
	else:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"target_idx": 0
		}
		print("ERROR - Machine '" + args["machine_id"] + "' has no TargettingDevice !")

	send_message(str.encode(json.dumps(state)))


def set_target_id(args):
	if flag_print_log:
		print(args["machine_id"] + " " + str(args["target_id"]))
	machine = main.destroyables_items[args["machine_id"]]
	td = machine.get_device("TargettingDevice")
	if td is not None:
		td.set_target_by_name(args["target_id"])
	else:
		print("ERROR - Machine '" + args["machine_id"] + "' has no TargettingDevice !")


def activate_autopilot(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		apctrl.activate()


def deactivate_autopilot(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		apctrl.deactivate()


def activate_IA(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	iactrl = machine.get_device("IAControlDevice")
	if iactrl is not None:
		iactrl.activate()


def deactivate_IA(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	iactrl = machine.get_device("IAControlDevice")
	if iactrl is not None:
		iactrl.deactivate()


def is_autopilot_activated(args):
	machine = main.destroyables_items[args["machine_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"autopilot": apctrl.is_activated()
		}
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
	else:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"autopilot": False
		}
	send_message(str.encode(json.dumps(state)))


def is_ia_activated(args):
	machine = main.destroyables_items[args["machine_id"]]
	iactrl = machine.get_device("IAControlDevice")
	if iactrl is not None:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"ia": iactrl.is_activated()
		}
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
	else:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"ia": False
		}
	send_message(str.encode(json.dumps(state)))


def is_user_control_activated(args):
	machine = main.destroyables_items[args["machine_id"]]
	uctrl = machine.get_device("UserControlDevice")
	if uctrl is not None:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"user": uctrl.is_activated()
		}
		if flag_print_log:
			print(args["machine_id"])
			print(str(state))
	else:
		state = {
			"timestamp": main.timestamp,
			"timestep": main.timestep,
			"user": False
		}
	send_message(str.encode(json.dumps(state)))


def activate_user_control(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	uctrl = machine.get_device("UserControlDevice")
	if uctrl is not None:
		uctrl.activate()


def deactivate_user_control(args):
	if flag_print_log:
		print(args["machine_id"])
	machine = main.destroyables_items[args["machine_id"]]
	uctrl = machine.get_device("UserControlDevice")
	if uctrl is not None:
		uctrl.deactivate()


def compute_next_timestep_physics(args):
	machine = main.destroyables_items[args["machine_id"]]
	physics_parameters = machine.get_physics_parameters()
	mat, physics_parameters = update_physics(machine.parent_node.GetTransform().GetWorld(), machine, physics_parameters, args["timestep"])
	v = physics_parameters["v_move"]
	physics_parameters["v_move"] = [v.x, v.y, v.z]
	mat_r0 = hg.GetRow(mat, 0)
	mat_r1 = hg.GetRow(mat, 1)
	mat_r2 = hg.GetRow(mat, 2)
	physics_parameters["matrix"] = [mat_r0.x, mat_r1.x, mat_r2.x,
									mat_r0.y, mat_r1.y, mat_r2.y, 
									mat_r0.z, mat_r1.z, mat_r2.z, 
									mat_r0.w, mat_r1.w, mat_r2.w]
	
	if flag_print_log:
		print(args["machine_id"])
		print(str(physics_parameters))
	send_message(str.encode(json.dumps(physics_parameters)))

# Aircraft


def get_plane_state(args):
	machine = main.destroyables_items[args["plane_id"]]
	h_spd, v_spd = machine.get_world_speed()

	gear = machine.get_device("Gear")
	apctrl = machine.get_device("AutopilotControlDevice")
	iactrl = machine.get_device("IAControlDevice")
	td = machine.get_device("TargettingDevice")

	if gear is not None:
		gear_activated = gear.activated
	else:
		gear_activated = False

	if apctrl is not None:
		autopilot_activated = apctrl.is_activated()
		autopilot_heading = apctrl.autopilot_heading
		autopilot_speed = apctrl.autopilot_speed
		autopilot_altitude = apctrl.autopilot_altitude
	else:
		autopilot_activated = False
		autopilot_heading = 0
		autopilot_speed = 0
		autopilot_altitude = 0

	if iactrl is not None:
		ia_activated = iactrl.is_activated()
	else:
		ia_activated = False

	if td is not None:
		target_id = td.get_target_name()
		target_locked = td.target_locked
	else:
		target_id = "- ! No TargettingDevice ! -"
		target_locked = False

	position = machine.get_position()
	rotation = machine.get_Euler()
	v_move = machine.get_move_vector()
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"position": [position.x, position.y, position.z],
		"Euler_angles": [rotation.x, rotation.y, rotation.z],
		"easy_steering": machine.flag_easy_steering,
		"health_level": machine.health_level,
		"destroyed": machine.flag_destroyed,
		"wreck": machine.wreck,
		"crashed": machine.flag_crashed,
		"active": machine.activated,
		"type": Destroyable_Machine.types_labels[machine.type],
		"nationality": machine.nationality,
		"thrust_level": machine.get_thrust_level(),
		"brake_level": machine.get_brake_level(),
		"flaps_level": machine.get_flaps_level(),
		"horizontal_speed": h_spd,
		"vertical_speed": v_spd,
		"linear_speed": machine.get_linear_speed(),
		"move_vector": [v_move.x, v_move.y, v_move.z],
		"linear_acceleration": machine.get_linear_acceleration(),
		"altitude": machine.get_altitude(),
		"heading": machine.get_heading(),
		"pitch_attitude": machine.get_pitch_attitude(),
		"roll_attitude": machine.get_roll_attitude(),
		"post_combustion": machine.post_combustion,
		"user_pitch_level": machine.get_pilot_pitch_level(),
		"user_roll_level": machine.get_pilot_roll_level(),
		"user_yaw_level": machine.get_pilot_yaw_level(),
		"gear": gear_activated,
		"ia": ia_activated,
		"autopilot": autopilot_activated,
		"autopilot_heading": autopilot_heading,
		"autopilot_speed": autopilot_speed,
		"autopilot_altitude": autopilot_altitude,
		"target_id": target_id,
		"target_locked": target_locked
	}
	if flag_print_log:
		print(args["plane_id"])
		print(str(state))
	send_message(str.encode(json.dumps(state)))


def get_planes_list(args):
	planes = []
	print("Get planes list")
	for dm in main.destroyables_list:
		if dm.type == Destroyable_Machine.TYPE_AIRCRAFT:
			print(dm.name)
			planes.append(dm.name)
	send_message(str.encode(json.dumps(planes)))


def record_plane_start_state(args):
	machine = main.destroyables_items[args["plane_id"]]
	machine.record_start_state()


def set_plane_linear_speed(args):
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_linear_speed(args["linear_speed"])


def reset_gear(args):
	machine = main.destroyables_items[args["plane_id"]]
	machine.flag_gear_deployed = args["gear_deployed"]
	gear = machine.get_device("Gear")
	if gear is not None:
		gear.reset()


def set_plane_thrust(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["thrust_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_thrust_level(args["thrust_level"])


def get_plane_thrust(args):
	machine = main.destroyables_items[args["plane_id"]]
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"thrust_level": machine.get_thrust_level()
	}
	if flag_print_log:
		print(args["plane_id"])
		print(str(state))
	send_message(str.encode(json.dumps(state)))


def activate_pc(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	machine.activate_post_combustion()


def deactivate_pc(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	machine.deactivate_post_combustion()


def set_plane_brake(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["brake_level"]))
		print(str(args["brake_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_brake_level(args["brake_level"])


def set_plane_flaps(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["flaps_level"]))
		print(str(args["flaps_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_flaps_level(args["flaps_level"])


def set_plane_pitch(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["pitch_level"]))
		print(str(args["pitch_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_pitch_level(args["pitch_level"])


def set_plane_roll(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["roll_level"]))
		print(str(args["roll_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_roll_level(args["roll_level"])


def set_plane_yaw(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["yaw_level"]))
		print(str(args["yaw_level"]))
	machine = main.destroyables_items[args["plane_id"]]
	machine.set_yaw_level(args["yaw_level"])


def stabilize_plane(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	machine.stabilize(True, True, True)


def deploy_gear(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	gear = machine.get_device("Gear")
	if gear is not None:
		gear.activate()


def retract_gear(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	gear = machine.get_device("Gear")
	if gear is not None:
		gear.deactivate()


def set_plane_autopilot_speed(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["ap_speed"]))
	machine = main.destroyables_items[args["plane_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		apctrl.set_autopilot_speed(args["ap_speed"])


def set_plane_autopilot_heading(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["ap_heading"]))
		print(str(args["ap_heading"]))
	machine = main.destroyables_items[args["plane_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		apctrl.set_autopilot_heading(args["ap_heading"])


def set_plane_autopilot_altitude(args):
	if flag_print_log:
		print(args["plane_id"] + " " + str(args["ap_altitude"]))
		print(str(args["ap_altitude"]))
	machine = main.destroyables_items[args["plane_id"]]
	apctrl = machine.get_device("AutopilotControlDevice")
	if apctrl is not None:
		apctrl.set_autopilot_altitude(args["ap_altitude"])


def activate_plane_easy_steering(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	machine.activate_easy_steering()


def deactivate_plane_easy_steering(args):
	if flag_print_log:
		print(args["plane_id"])
	machine = main.destroyables_items[args["plane_id"]]
	machine.deactivate_easy_steering()


# Missile launchers

def get_missile_launchers_list(args):
	missile_launchers = []
	print("Get missile launchers list")
	for dm in main.destroyables_list:
		print(dm.name)
		if dm.type == Destroyable_Machine.TYPE_MISSILE_LAUNCHER:
			missile_launchers.append(dm.name)
	send_message(str.encode(json.dumps(missile_launchers)))


def get_missile_launcher_state(args):
	machine = main.destroyables_items[args["machine_id"]]
	position = machine.get_position()
	rotation = machine.get_Euler()
	td = machine.get_device("TargettingDevice")
	if td is not None:
		target_id = td.get_target_name()
		target_locked = td.target_locked
	else:
		target_id = "- ! No TargettingDevice ! -"
		target_locked = False
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"position": [position.x, position.y, position.z],
		"Euler_angles": [rotation.x, rotation.y, rotation.z],
		"health_level": machine.health_level,
		"destroyed": machine.flag_destroyed,
		"wreck": machine.wreck,
		"active": machine.activated,
		"type": Destroyable_Machine.types_labels[machine.type],
		"nationality": machine.nationality,
		"altitude": machine.get_altitude(),
		"heading": machine.get_heading(),
		"target_id": target_id,
		"target_locked": target_locked
	}
	print("State OK")
	if flag_print_log:
		print(args["machine_id"])
		print(str(state))
	send_message(str.encode(json.dumps(state)))


# Missiles

def get_missiles_list(args):
	print("Get missiles list")
	missiles = []
	for dm in main.destroyables_list:
		if dm.type == Destroyable_Machine.TYPE_MISSILE:
			print(dm.name)
			missiles.append(dm.name)
	send_message(str.encode(json.dumps(missiles)))


def get_missile_state(args):
	machine = main.destroyables_items[args["missile_id"]]
	position = machine.get_position()
	rotation = machine.get_Euler()
	v_move = machine.get_move_vector()
	h_spd, v_spd = machine.get_world_speed()
	state = {
		"timestamp": main.timestamp,
		"timestep": main.timestep,
		"type": Destroyable_Machine.types_labels[machine.type],
		"position": [position.x, position.y, position.z],
		"Euler_angles": [rotation.x, rotation.y, rotation.z],
		"move_vector": [v_move.x, v_move.y, v_move.z],
		"destroyed": machine.flag_destroyed,
		"wreck": machine.wreck,
		"crashed": machine.flag_crashed,
		"active": machine.activated,
		"nationality": machine.nationality,
		"altitude": machine.get_altitude(),
		"heading": machine.get_heading(),
		"pitch_attitude": machine.get_pitch_attitude(),
		"roll_attitude": machine.get_roll_attitude(),
		"horizontal_speed": h_spd,
		"vertical_speed": v_spd,
		"linear_speed": machine.get_linear_speed(),
		"target_id": machine.get_target_id(),
		"life_delay": machine.life_delay,
		"life_time": machine.life_cptr,
		"thrust_force": machine.f_thrust,
		"angular_frictions": [machine.angular_frictions.x, machine.angular_frictions.y, machine.angular_frictions.z],
		"drag_coefficients": [machine.drag_coeff.x, machine.drag_coeff.y, machine.drag_coeff.z]
		}
	if flag_print_log:
		print(args["missile_id"])
		print(str(state))
	end_message(str.encode(json.dumps(state)))


def set_missile_life_delay(args):
	missile = main.destroyables_items[args["missile_id"]]
	missile.set_life_delay(args["life_delay"])


def set_missile_target(args):
	missile = main.destroyables_items[args["missile_id"]]
	missile.set_target_by_name(args["target_id"])


def get_missile_targets_list(args):
	missile = main.destroyables_items[args["missile_id"]]
	targets = missile.get_valid_targets_list()
	targets_ids = ["-None-"]
	for t in targets:
		targets_ids.append(t.name)
	send_message(str.encode(json.dumps(targets_ids)))

def set_missile_thrust_force(args):
	missile = main.destroyables_items[args["missile_id"]]
	missile.set_thrust_force(args["thrust_force"])

def set_missile_angular_frictions(args):
	missile = main.destroyables_items[args["missile_id"]]
	missile.set_angular_friction(args["angular_frictions"][0], args["angular_frictions"][1], args["angular_frictions"][2] )

def set_missile_drag_coefficients(args):
	missile = main.destroyables_items[args["missile_id"]]
	missile.set_drag_coefficients(args["drag_coeff"][0], args["drag_coeff"][1], args["drag_coeff"][2] )



class Miuss_Parameters:

    model_name = "Miuss"
    instance_scene_name = "machines/mius/miuss.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 30
        self.thrust_force = 20
        self.post_combution_force = self.thrust_force
        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
        self.speed_ceiling = 3000  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 60000
        self.max_altitude = 50000
        self.gear_height = 1.75
        self.bottom_height = 0.85

        # Weapons configuration:
        self.missiles_config = ["Mica", "Meteor", "AIM_SL", "AIM_SL", "Meteor", "Mica"]

        # Mobile parts:
        self.mobile_parts_definitions = [
            ["aileron_left", 45, -45, 0, "dummy_flap_left", "X"],
            ["aileron_right", 45, -45, 0, "dummy_flap_right", "X"],
            ["elevator", -20, 20, 0, "dummy_elevator", "X"]
        ]


class Miuss(Aircraft, Miuss_Parameters):

    @classmethod
    def init(cls, scene):
        print("Miuss class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):

        self.gear_anim_play = None

        Aircraft.__init__(self, name, Miuss_Parameters.model_name, scene, scene_physics, pipeline_ressource, Miuss.instance_scene_name, nationality, start_pos, start_rot)
        Miuss_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        gear = Gear("Gear", self, scene, self.get_animation("gear_open"), self.get_animation("gear_close"))
        gear.gear_moving_delay = 3
        self.add_device(gear)
        md = self.get_device("MissilesDevice")
        if md is not None:
            md.flag_hide_fitted_missiles = False
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):
        self.parts["elevator"]["level"] = -self.angular_levels.x
        self.parts["aileron_left"]["level"] = self.angular_levels.z
        self.parts["aileron_right"]["level"] = -self.angular_levels.z

        Aircraft.update_mobile_parts(self, dts)
class TFX_Parameters:

    model_name = "TFX"
    instance_scene_name = "machines/tfx/TFX.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 45
        self.thrust_force = 20
        self.post_combution_force = self.thrust_force
        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
        self.speed_ceiling = 2500  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 25000
        self.max_altitude = 30000
        self.gear_height = 1.95227
        self.bottom_height = 0.748589

        # Weapons configuration:
        self.missiles_config = ["AIM_SL", "AIM_SL", "AIM_SL", "AIM_SL"]

        # Mobile parts:
        self.mobile_parts_definitions = [
            ["aileron_left", -45, 45, 0, "dummy_aileron_left", "Z"],
            ["aileron_right", -45, 45, 0, "dummy_aileron_right", "Z"],
            ["elevator", -11, 11, 0, "dummy_elevator", "X"],
            ["rudder_left", -45, 45, None, "dummy_rudder_left", "Z"],
            ["rudder_right", -45, 45, None, "dummy_rudder_right", "Z"]
        ]


class TFX(Aircraft, TFX_Parameters):

    @classmethod
    def init(cls, scene):
        print("TFX class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):

        self.gear_anim_play = None

        Aircraft.__init__(self, name, TFX_Parameters.model_name, scene, scene_physics, pipeline_ressource, TFX.instance_scene_name, nationality, start_pos, start_rot)
        TFX_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        gear = Gear("Gear", self, scene, self.get_animation("gear_open"), self.get_animation("gear_close"))
        gear.gear_moving_delay = 3
        self.add_device(gear)
        md = self.get_device("MissilesDevice")
        if md is not None:
            md.flag_hide_fitted_missiles = True
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):
        self.parts["elevator"]["level"] = -self.angular_levels.x
        self.parts["aileron_left"]["level"] = self.angular_levels.z
        self.parts["aileron_right"]["level"] = -self.angular_levels.z
        self.parts["rudder_left"]["level"] = -self.angular_levels.y
        self.parts["rudder_right"]["level"] = -self.angular_levels.y

        Aircraft.update_mobile_parts(self, dts)
class F16_Parameters:

    model_name = "F16"
    instance_scene_name = "machines/f16/F16_rigged.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 35
        self.thrust_force = 15
        self.post_combution_force = self.thrust_force

        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)

        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.speed_ceiling = 1750  # maneuverability is not guaranteed beyond this speed !

        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll

        self.mobile_parts_definitions = [
            ["aileron_left", -45, 45, 0, "dummy_flap_left", "X"],
            ["aileron_right", -45, 45, 0, "dummy_flap_right", "X"],
            ["elevator_left", -15, 15, 0, "dummy_elevator_left", "X"],
            ["elevator_right", -15, 15, 0, "dummy_elevator_right", "X"],
            ["rudder", -45, 45, 0, "dummy_rudder", "Z"]
        ]

        self.max_safe_altitude = 15700
        self.max_altitude = 25700
        self.gear_height = 2.504 * 0.9
        self.bottom_height = 1.3 * 0.9

        # Weapons configuration:
        self.missiles_config = ["AIM_SL", "AIM_SL", "Karaoke", "Karaoke", "Karaoke", "Karaoke", "CFT", "CFT", "Karaoke", "Karaoke", "Karaoke", "Karaoke"]


class F16(Aircraft, F16_Parameters):

    @classmethod
    def init(cls, scene):
        print("F16 class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):
        self.gear_anim_play = None
        Aircraft.__init__(self, name, F16_Parameters.model_name, scene, scene_physics, pipeline_ressource, F16.instance_scene_name, nationality, start_pos, start_rot)
        F16_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        self.add_device(Gear("Gear", self, scene, self.get_animation("gear_open"), self.get_animation("gear_close")))
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):

        self.parts["aileron_left"]["level"] = -self.angular_levels.z
        self.parts["aileron_right"]["level"] = self.angular_levels.z
        self.parts["elevator_left"]["level"] = self.angular_levels.x
        self.parts["elevator_right"]["level"] = self.angular_levels.x
        self.parts["rudder"]["level"] = self.angular_levels.y
        Aircraft.update_mobile_parts(self, dts)
class F14_Parameters():
    model_name = "F14"
    instance_scene_name = "machines/aircraft/aircraft_blend.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 40
        self.thrust_force = 10
        self.post_combution_force = self.thrust_force / 2
        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
        self.speed_ceiling = 1750  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 15700
        self.max_altitude = 25700

        self.gear_height = 3
        self.bottom_height = 2

        # Weapons configuration:
        self.missiles_config = ["Sidewinder", "Sidewinder", "Sidewinder", "Sidewinder"]

        # Mobile parts:

        self.wings_thresholds = hg.Vec2(500, 750)
        self.wings_level = 0
        self.wings_geometry_gain_friction = -0.0001

        self.mobile_parts_definitions = [
            ["aileron_left", -45, 45, 0, "dummy_aircraft_aileron_l", "X"],
            ["aileron_right", -45, 45, 0, "dummy_aircraft_aileron_r", "X"],
            ["elevator_left", -15, 15, 0, "dummy_aircraft_elevator_changepitch_l", "X"],
            ["elevator_right", -15, 15, 0, "dummy_aircraft_elevator_changepitch_r", "X"],
            ["rudder_left", -45 + 180, 45 + 180, 180, "aircraft_rudder_changeyaw_l", "Y"],
            ["rudder_right", -45, 45, 0, "aircraft_rudder_changeyaw_r", "Y"],
            ["wing_left", -45, 0, 0, "dummy_aircraft_configurable_wing_l", "Y"],
            ["wing_right", 0, 45, 0, "dummy_aircraft_configurable_wing_r", "Y"]
        ]

# Main Aircraft, with physics interactions

class F14(Aircraft, F14_Parameters):

    @classmethod
    def init(cls, scene):
        print("F14 class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):
        Aircraft.__init__(self, name, F14_Parameters.model_name, scene, scene_physics, pipeline_ressource, F14_Parameters.instance_scene_name, nationality, start_pos, start_rot)
        F14_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        self.add_device(Gear("Gear", self)) #, scene, self.get_animation("gear_open"), self.get_animation("gear_close")))
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):
        self.parts["aileron_left"]["level"] = -self.angular_levels.z
        self.parts["aileron_right"]["level"] = -self.angular_levels.z
        self.parts["elevator_left"]["level"] = -self.angular_levels.x
        self.parts["elevator_right"]["level"] = -self.angular_levels.x
        self.parts["rudder_left"]["level"] = self.angular_levels.y
        self.parts["rudder_right"]["level"] = -self.angular_levels.y
        self.set_wings_level(self.get_linear_speed())
        Aircraft.update_mobile_parts(self, dts)

    def set_wings_level(self, frontal_speed):
        value = max(min((frontal_speed * 3.6 - self.wings_thresholds.x) / (self.wings_thresholds.y - self.wings_thresholds.x), 1), 0)
        self.wings_level = min(max(value, 0), 1)
        self.parts["wing_left"]["level"] = -value
        self.parts["wing_right"]["level"] = value

    def compute_z_drag(self):
        return Aircraft.compute_z_drag(self) + self.wings_geometry_gain_friction * self.wings_level
class F14_2_Parameters():

    model_name = "F14_2"
    instance_scene_name = "machines/ennemy_aircraft/ennemyaircraft_blend.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 40
        self.thrust_force = 10
        self.post_combution_force = self.thrust_force / 2
        self.drag_coeff = hg.Vec3(0.033, 0.06666, 0.0002)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
        self.speed_ceiling = 1750  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 15700
        self.max_altitude = 25700

        self.gear_height = 3
        self.bottom_height = 2

        # Weapons configuration:
        self.missiles_config = ["Sidewinder", "Sidewinder", "Sidewinder", "Sidewinder"]

        # Mobile parts:

        self.wings_thresholds = hg.Vec2(500, 750)
        self.wings_level = 0
        self.wings_geometry_gain_friction = -0.0001

        self.mobile_parts_definitions = [
            ["aileron_left", -45, 45, 0, "dummy_ennemyaircraft_aileron_l", "X"],
            ["aileron_right", -45, 45, 0, "dummy_ennemyaircraft_aileron_r", "X"],
            ["elevator_left", -15, 15, 0, "dummy_ennemyaircraft_elevator_changepitch_l", "X"],
            ["elevator_right", -15, 15, 0, "dummy_ennemyaircraft_elevator_changepitch_r", "X"],
            ["rudder_left", -45 + 180, 45 + 180, 180, "ennemyaircraft_rudder_changeyaw_l", "Y"],
            ["rudder_right", -45, 45, 0, "ennemyaircraft_rudder_changeyaw_r", "Y"],
            ["wing_left", -45, 0, 0, "dummy_ennemyaircraft_configurable_wing_l", "Y"],
            ["wing_right", 0, 45, 0, "dummy_ennemyaircraft_configurable_wing_r", "Y"]
        ]


class F14_2(Aircraft, F14_2_Parameters):


    @classmethod
    def init(cls, scene):
        print("F14_2 class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):
        Aircraft.__init__(self, name, F14_2_Parameters.model_name, scene, scene_physics, pipeline_ressource, F14_2.instance_scene_name, nationality, start_pos, start_rot)
        F14_2_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        self.add_device(Gear("Gear", self))
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):
        self.parts["aileron_left"]["level"] = -self.angular_levels.z
        self.parts["aileron_right"]["level"] = -self.angular_levels.z
        self.parts["elevator_left"]["level"] = -self.angular_levels.x
        self.parts["elevator_right"]["level"] = -self.angular_levels.x
        self.parts["rudder_left"]["level"] = self.angular_levels.y
        self.parts["rudder_right"]["level"] = -self.angular_levels.y
        self.set_wings_level(self.get_linear_speed())
        Aircraft.update_mobile_parts(self, dts)

    def set_wings_level(self, frontal_speed):
        value = max(min((frontal_speed * 3.6 - self.wings_thresholds.x) / (self.wings_thresholds.y - self.wings_thresholds.x), 1), 0)
        self.wings_level = min(max(value, 0), 1)
        self.parts["wing_left"]["level"] = -value
        self.parts["wing_right"]["level"] = value

    def compute_z_drag(self):
        return Aircraft.compute_z_drag(self) + self.wings_geometry_gain_friction * self.wings_level
class Rafale_Parameters:

    model_name = "Rafale"
    instance_scene_name = "machines/rafale/rafale_rigged.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 30

        self.thrust_force = 15
        self.post_combution_force = self.thrust_force / 2
        self.drag_coeff = hg.Vec3(0.043, 0.07666, 0.0003)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000165, 0.000115, 0.000255)  # pitch, yaw, roll
        self.speed_ceiling = 2200  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)
        self.max_safe_altitude = 15240
        self.max_altitude = 25240 * 4

        self.gear_height = 2.28
        self.bottom_height = 0.78

        # Weapons configuration:
        self.missiles_config = ["Mica", "Meteor", "Meteor", "Meteor", "Meteor", "Mica"]

        # Mobile parts:

        self.mobile_parts_definitions = [
            ["aileron_left", -20, 20, 0, "dummy_rafale_wing_flap_l", "X"],
            ["aileron_right", -20, 20, 0, "dummy_rafale_wing_flap_r", "X"],
            ["elevator_left", -45, 45, 0, "dummy_rafale_elevator_l", "X"],
            ["elevator_right", -45, 45, 0, "dummy_rafale_elevator_r", "X"],
            ["rudder", -30, 30, 0, "rafale_rudder", "Y"]
        ]


class Rafale(Aircraft, Rafale_Parameters):

    @classmethod
    def init(cls, scene):
        print("Rafale class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):

        self.gear_anim_play = None

        Aircraft.__init__(self, name, Rafale_Parameters.model_name, scene, scene_physics, pipeline_ressource, Rafale.instance_scene_name, nationality, start_pos, start_rot)
        Rafale_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        self.add_device(Gear("Gear", self, scene, self.get_animation("gear_open"), self.get_animation("gear_fr_close")))
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):

        self.parts["aileron_left"]["level"] = -self.angular_levels.z
        self.parts["aileron_right"]["level"] = self.angular_levels.z
        self.parts["elevator_left"]["level"] = -self.angular_levels.x
        self.parts["elevator_right"]["level"] = -self.angular_levels.x
        self.parts["rudder"]["level"] = -self.angular_levels.y
        Aircraft.update_mobile_parts(self, dts)
class Eurofighter_Parameters:

    model_name = "Eurofighter"
    instance_scene_name = "machines/eurofighter/eurofighter_anim.scn"

    def __init__(self):
        # Aircraft constants:
        self.camera_track_distance = 30
        self.thrust_force = 13
        self.post_combution_force = self.thrust_force / 1.5
        self.drag_coeff = hg.Vec3(0.043, 0.07666, 0.0003)
        self.wings_lift = 0.0005
        self.brake_drag = 0.006
        self.flaps_lift = 0.0025
        self.flaps_drag = 0.002
        self.angular_frictions = hg.Vec3(0.000190, 0.000170, 0.000300)  # pitch, yaw, roll
        self.speed_ceiling = 2500  # maneuverability is not guaranteed beyond this speed !
        self.angular_levels_inertias = hg.Vec3(3, 3, 3)

        self.max_safe_altitude = 16800
        self.max_altitude = 26800

        self.gear_height = 2.02
        self.bottom_height = 1.125

        # Weapons configuration:
        self.missiles_config = ["Meteor", "Mica", "Mica", "Mica", "Mica", "Meteor"]

        # Mobile parts:

        self.mobile_parts_definitions = [
            ["aileron_left", -20, 20, 0, "dummy_wing_flap_l", "X"],
            ["aileron_right", -20, 20, 0, "dummy_wing_flap_r", "X"],
            ["elevator", -45, 45, 0, "dummy_elevator", "X"],
            ["rudder", -30, 30, 0, "rudder", "Y"],
            ["brake_flap", 0, 33.1, 0, "dummy_brake_flap", "X"],
            ["brake_handle", -47.4522, 0, 0, "dummy_brake_handle", "X"]
            ]

        self.brake_flap_a = 1
        self.brake_flap_v = 0
        self.brake_flap_vmax = 1
        self.current_brake_flap_level = 0


class Eurofighter(Aircraft, Eurofighter_Parameters):


    @classmethod
    def init(cls, scene):
        print("Eurofighter class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_pos, start_rot):

        self.gear_anim_play = None

        Aircraft.__init__(self, name, Eurofighter_Parameters.model_name, scene, scene_physics, pipeline_ressource, Eurofighter_Parameters.instance_scene_name, nationality, start_pos, start_rot)
        Eurofighter_Parameters.__init__(self)
        self.define_mobile_parts(self.mobile_parts_definitions)
        self.add_device(Gear("Gear", self, scene, self.get_animation("gear_open"), self.get_animation("gear_close")))
        self.setup_bounds_positions()

    def update_mobile_parts(self, dts):

        self.parts["aileron_left"]["level"] = self.angular_levels.z
        self.parts["aileron_right"]["level"] = -self.angular_levels.z
        self.parts["elevator"]["level"] = self.angular_levels.x
        self.parts["rudder"]["level"] = -self.angular_levels.y

        if self.current_brake_flap_level < self.brake_level:
            self.brake_flap_v = max(-self.brake_flap_vmax, min(self.brake_flap_vmax, self.brake_flap_v + self.brake_flap_a * dts))
        else:
            self.brake_flap_v = min(self.brake_flap_vmax, max(-self.brake_flap_vmax, self.brake_flap_v - self.brake_flap_a * dts))
        m = self.current_brake_flap_level
        self.current_brake_flap_level = self.current_brake_flap_level + self.brake_flap_v * dts
        if (m < self.brake_level < self.current_brake_flap_level) or (m > self.brake_level > self.current_brake_flap_level):
            self.current_brake_flap_level = self.brake_level
            self.brake_flap_v = 0
        if self.current_brake_flap_level < 0:
            self.current_brake_flap_level = 0
            self.brake_flap_v = 0
        if self.current_brake_flap_level > 1:
            self.current_brake_flap_level = 1
            self.brake_flap_v = 0

        self.parts["brake_flap"]["level"] = pow(self.current_brake_flap_level, 0.4)
        self.parts["brake_handle"]["level"] = -self.current_brake_flap_level

        Aircraft.update_mobile_parts(self, dts)
class MissileLauncherS400(LandVehicle):
	model_name = "Missile_Launcher"
	instance_scene_name = "machines/missile_launcher/missile_launcher_exp.scn"

	def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality, start_position, start_rotation):
		LandVehicle.__init__(self, name, MissileLauncherS400.model_name, scene, scene_physics, pipeline_ressource, MissileLauncherS400.instance_scene_name, nationality, start_position, start_rotation)
		self.type = Destroyable_Machine.TYPE_MISSILE_LAUNCHER

		self.brake_level = 1

		self.add_device(MissileLauncherUserControlDevice("UserControlDevice", self, "scripts/missile_launcher_user_inputs_mapping.json"))

		td = TargettingDevice("TargettingDevice", self, True)
		self.add_device(td)
		td.set_target_lock_range(2000, 20000)
		td.flag_front_lock_cone = False

		self.missiles_config = ["S400", "S400", "S400", "S400"]
		self.missiles_slots = self.get_missiles_slots()
		self.add_device(MissilesDevice("MissilesDevice", self, self.missiles_slots))

		self.plateform = None

		# Views parameters
		self.camera_track_distance = 100
		self.camera_follow_distance = 100
		self.camera_tactical_distance = 100
		self.camera_tactical_min_altitude = 1


	def destroy(self):

		md = self.get_device("MissilesDevice")
		if md is not None:
			md.destroy()

		self.destroy_nodes()
		self.flag_destroyed = True

	def set_platform(self, plateform: hg.Node):
		self.plateform = plateform
		self.start_position = self.plateform.GetTransform().GetPos()
		self.start_rotation = self.plateform.GetTransform().GetRot()
		self.parent_node.GetTransform().SetPos(self.start_position)
		self.parent_node.GetTransform().SetRot(self.start_rotation)

	# =========================== Missiles

	def get_missiles_slots(self):
		return self.get_slots("missile_slot")

	def get_num_missiles_slots(self):
		return len(self.missiles_slots)

	def rearm(self):
		self.set_health_level(1)
		md = self.get_device("MissilesDevice")
		if md is not None:
			md.rearm()

	#=========================== Physics


	def update_kinetics(self, dts):
		if self.activated:
			if self.custom_matrix is not None:
				matrix = self.custom_matrix
			else:
				matrix = self.get_parent_node().GetTransform().GetWorld()
			if self.custom_v_move is not None:
				v_move = self.custom_v_move
			else:
				v_move = self.v_move

			if not self.flag_crashed:
				self.v_move = v_move

			pos = hg.GetT(matrix)

			if not self.flag_custom_physics_mode:
				pos += self.v_move * dts

			# Collisions
			if self.plateform is not None:
				p_pos = self.plateform.GetTransform().GetPos()
				alt = p_pos.y
			else:
				alt = get_terrain_altitude(pos)

			if pos.y < alt:
				pos.y += (alt - pos.y) * 0.1 * 60 * dts
				if self.v_move.y < 0: self.v_move.y *= pow(0.8, 60 * dts)
				# b = min(1, self.brake_level + (1 - health_wreck_factor))
				b = self.brake_level
				self.v_move *= ((b * pow(0.98, 60 * dts)) + (1 - b))

			else:
				self.v_move += F_gravity * dts

			self.parent_node.GetTransform().SetPos(pos)

			self.update_devices(dts)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			self.update_mobile_parts(dts)
			self.update_feedbacks(dts)
class Sidewinder(Missile):
    model_name = "Sidewinder"
    instance_scene_name = "weaponry/missile_sidewinder.scn"

    @classmethod
    def init(cls, scene):
        print("Sidewinder missile class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, Sidewinder.model_name, nationality, scene, scene_physics, pipeline_ressource, Sidewinder.instance_scene_name)

        self.f_thrust = 100
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.00008, 0.00008, 0.00008)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 20
        self.smoke_delay = 1

    def get_hit_damages(self):
        return uniform(0.30, 0.40)
class Meteor(Missile):
    model_name = "Meteor"
    instance_scene_name = "weaponry/missile_meteor.scn"

    @classmethod
    def init(cls, scene):
        print("Meteor missile class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, Meteor.model_name, nationality, scene, scene_physics, pipeline_ressource, Meteor.instance_scene_name)

        self.f_thrust = 80
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.00005, 0.00005, 0.00005)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 40
        self.smoke_delay = 1.5

    def get_hit_damages(self):
        return uniform(0.40, 0.60)
class Mica(Missile):
    model_name = "Mica"
    instance_scene_name = "weaponry/missile_mica.scn"

    @classmethod
    def init(cls, scene):
        print("Mica missile class init")


    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, Mica.model_name, nationality, scene, scene_physics, pipeline_ressource, Mica.instance_scene_name)

        self.f_thrust = 150
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.00014, 0.00014, 0.00014)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 15
        self.smoke_delay = 1

    def get_hit_damages(self):
        return uniform(0.20, 0.30)
class AIM_SL(Missile):
    model_name = "AIM_SL"
    instance_scene_name = "weaponry/missile_aim_sl.scn"

    @classmethod
    def init(cls, scene):
        print("AIM-SL missile class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, AIM_SL.model_name, nationality, scene, scene_physics, pipeline_ressource, AIM_SL.instance_scene_name)

        self.f_thrust = 120
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.00008, 0.00008, 0.00008)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 20
        self.smoke_delay = 1

    def get_hit_damages(self):
        return uniform(0.25, 0.35)
class Karaoke(Missile):
    model_name = "Karaoke"
    instance_scene_name = "weaponry/missile_karaoke.scn"

    @classmethod
    def init(cls, scene):
        print("Karaoke missile class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, Karaoke.model_name, nationality, scene, scene_physics, pipeline_ressource, Karaoke.instance_scene_name)

        self.f_thrust = 70
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.00005, 0.00005, 0.00005)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 35
        self.smoke_delay = 1.5

    def get_hit_damages(self):
        return uniform(0.50, 0.70)
class S400(Missile):
    model_name = "S400"
    instance_scene_name = "machines/S400/S400.scn"

    @classmethod
    def init(cls, scene):
        print("S400 missile class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, S400.model_name, nationality, scene, scene_physics, pipeline_ressource, S400.instance_scene_name)

        self.f_thrust = 60
        self.smoke_parts_distance = 1.44374
        self.angular_frictions = hg.Vec3(0.000025, 0.000025, 0.000025)  # pitch, yaw, roll
        self.drag_coeff = hg.Vec3(0.37, 0.37, 0.0003)
        self.life_delay = 100
        self.smoke_delay = 1.5

    def get_hit_damages(self):
        return uniform(0.2, 0.3)
class CFT(Missile):
    model_name = "CFT"
    instance_scene_name = "weaponry/fuel_cft.scn"

    @classmethod
    def init(cls, scene):
        print("CFT class init")

    def __init__(self, name, scene, scene_physics, pipeline_ressource: hg.PipelineResources, nationality):
        Missile.__init__(self, name, CFT.model_name, nationality, scene, scene_physics, pipeline_ressource, CFT.instance_scene_name)

        self.flag_armed = False
class SmartCamera:
	TYPE_FOLLOW = 1
	TYPE_TRACKING = 2
	TYPE_SATELLITE = 3
	TYPE_FPS = 4
	TYPE_CINEMATIC = 5
	TYPE_FIX = 6
	TYPE_TACTICAL = 7

	def __init__(self, cam_type=TYPE_FOLLOW, keyboard=None, mouse=None):

		self.flag_hovering_gui = False

		self.flag_fix_mouse_controls_rotation = True  # True = FIX camera rotation controlled with mouse
		self.flag_reseting_rotation = False

		self.type = cam_type

		self.keyboard = keyboard
		self.mouse = mouse

		self.fix_pos = None
		self.fix_rot = None
		self.fix_rot_inertia = 0.1

		self.fps_pos = None
		self.fps_rot = None
		self.fps_rot_inertia = 0.1
		self.fps_pos_inertia = 0.1

		self.track_position = None  # Vec3
		self.track_orientation = None  # Mat3 - Rotation inertia matrix

		self.pos_inertia = 0.2
		self.rot_inertia = 0.07

		self.follow_inertia = 0.01
		self.follow_distance = 40
		self.lateral_rot = 20

		self.target_node = None
		self.target_point = None
		self.target_matrix = None

		self.satellite_view_size = 100
		self.satellite_view_size_inertia = 0.7
		self.satellite_view_ratio = 16 / 9

		self.camera_move = hg.Vec3(0, 0, 0)  # Translation in frame

		self.noise_x = Temporal_Perlin_Noise(0.1446)
		self.noise_y = Temporal_Perlin_Noise(0.1017)
		self.noise_z = Temporal_Perlin_Noise(0.250314)
		self.noise_level = 0.1

		self.back_view = {"position": hg.Vec3(0, 4, -20),
						  "orientation": hg.Mat3(hg.Vec3(1, 0, 0), hg.Vec3(0, 1, 0), hg.Vec3(0, 0, 1)),
						  "pos_inertia": 0.2, "rot_inertia": 0.07}

		self.front_view = {"position": hg.Vec3(0, 4, 40),
						   "orientation": hg.Mat3(hg.Vec3(-1, 0, 0), hg.Vec3(0, 1, 0), hg.Vec3(0, 0, -1)),
						   "pos_inertia": 0.9, "rot_inertia": 0.05}

		self.right_view = {"position": hg.Vec3(-40, 4, 0),
						   "orientation": hg.Mat3(hg.Vec3(0, 0, -1), hg.Vec3(0, 1, 0), hg.Vec3(1, 0, 0)),
						   "pos_inertia": 0.9, "rot_inertia": 0.05}

		self.left_view = {"position": hg.Vec3(40, 4, 0),
						  "orientation": hg.Mat3(hg.Vec3(0, 0, 1), hg.Vec3(0, 1, 0), hg.Vec3(-1, 0, 0)),
						  "pos_inertia": 0.9, "rot_inertia": 0.05}

		self.top_view = {"position": hg.Vec3(0, 50, 0),
						 "orientation": hg.Mat3(hg.Vec3(1, 0, 0), hg.Vec3(0, 0, 1), hg.Vec3(0, -1, 0)),
						 "pos_inertia": 0.9, "rot_inertia": 0.05}

		self.views = {
			"back": self.back_view,
			"front": self.front_view,
			"right": self.right_view,
			"left": self.left_view,
			"top": self.top_view
		}

		self.current_view = ""
		self.set_track_view("back")

		self.cinematic_timer = 0
		self.keyframes = []

		# Tactical:

		self.subject_node = None
		self.tactical_subject_distance = 40
		self.minimal_tactical_altitude = 10
		self.projectil_node = None
		self.tactical_pos_inertia = 0.75

	# ----------- Call setup at view beginning:

	def setup(self, cam_type, camera: hg.Node, target_node: hg.Node = None, target_point: hg.Vec3 = None, target_matrix: hg.Mat3 = None):
		self.type = cam_type
		camera.GetTransform().ClearParent()
		if self.type == SmartCamera.TYPE_SATELLITE:
			self.setup_satellite_camera(camera)
		self.target_node = target_node
		if target_node is not None:
			w = target_node.GetTransform().GetWorld()
			if target_point is None:
				self.target_point = hg.GetT(w)
			else:
				self.target_point = target_point
			if target_matrix is None:
				self.target_matrix = hg.RotationMat3(hg.GetR(w))
			else:
				self.target_matrix = target_matrix

		if self.type == SmartCamera.TYPE_FOLLOW:
			self.update_follow_direction(camera)
			self.update_follow_translation(camera, 0, True)
		elif self.type == SmartCamera.TYPE_TRACKING:
			self.update_track_direction(camera, 0, 0)
			self.update_track_translation(camera)
		elif self.type == SmartCamera.TYPE_SATELLITE:
			self.update_satellite_camera(camera, 0, True)
		elif self.type == SmartCamera.TYPE_FPS:
			self.fps_pos = camera.GetTransform().GetPos()
			self.fps_rot = camera.GetTransform().GetRot()
			self.update_fps_camera(camera, 0)

		elif self.type == SmartCamera.TYPE_CINEMATIC:
			self.cinematic_timer = 0
			self.update_cinematic_camera(camera, 0)

		elif self.type == SmartCamera.TYPE_FIX:
			trans = camera.GetTransform()
			trans.SetPos(hg.Vec3(0, 0, 0))
			trans.SetRot(hg.Vec3(0, 0, 0))
			camera.GetTransform().SetParent(self.target_node)
			self.fix_rot = hg.Vec3(0, 0, 0) # FIX camera rotation reset to 0
			self.fix_pos = camera.GetTransform().GetPos()
			self.update_fix_camera(camera, 0)

	def setup_tactical(self,camera: hg.Node, subject_node: hg.Node, target_node:hg.Node, projectil_node:Destroyable_Machine = None ):
		self.type = SmartCamera.TYPE_TACTICAL
		camera.GetTransform().ClearParent()
		self.subject_node = subject_node
		self.target_node = target_node
		self.projectil_node = projectil_node
		self.target_point = self.compute_tactical_view_center(camera.GetTransform().GetPos())
		self.update_tactical_camera(camera, 0, True)

	def update(self, camera: hg.Camera, dts, noise_level=0):
		if self.type == SmartCamera.TYPE_FOLLOW:
			self.update_camera_follow(camera, dts)
		elif self.type == SmartCamera.TYPE_TRACKING:
			self.update_camera_tracking(camera, dts, noise_level)
		elif self.type == SmartCamera.TYPE_SATELLITE:
			self.update_satellite_camera(camera, dts)
		elif self.type == SmartCamera.TYPE_FPS:
			self.update_fps_camera(camera, dts)
		elif self.type == SmartCamera.TYPE_CINEMATIC:
			self.update_cinematic_camera(camera, dts)
		elif self.type == SmartCamera.TYPE_FIX:
			self.update_fix_camera(camera, dts, noise_level)
		elif self.type == SmartCamera.TYPE_TACTICAL:
			self.update_tactical_camera(camera, dts)

	def update_target_point(self, dts):
		v = self.target_node.GetTransform().GetPos() - self.target_point
		self.target_point += v * self.pos_inertia * dts * 60
		mat_n = hg.TransformationMat4(self.target_node.GetTransform().GetPos(), self.target_node.GetTransform().GetRot())
		rz = hg.Cross(hg.GetZ(self.target_matrix), hg.GetZ(mat_n))
		ry = hg.Cross(hg.GetY(self.target_matrix), hg.GetY(mat_n))
		mr = rz + ry
		le = hg.Len(mr)
		if le > 0.001:
			self.target_matrix = MathsSupp.rotate_matrix(self.target_matrix, hg.Normalize(mr), le * self.rot_inertia * dts * 60)

	# ============== Camera fix =====================
	def enable_mouse_controls_fix_rotation(self):
		self.flag_fix_mouse_controls_rotation = True

	def disable_mouse_controls_fix_rotation(self):
		self.flag_fix_mouse_controls_rotation = False


	def update_fix_camera(self, camera, dts, noise_level=0):
		f = radians(noise_level)
		rot = hg.Vec3(self.noise_x.temporal_Perlin_noise(dts) * f, self.noise_y.temporal_Perlin_noise(dts) * f, self.noise_z.temporal_Perlin_noise(dts) * f)

		if self.flag_fix_mouse_controls_rotation:

			if self.flag_reseting_rotation:
				self.fix_rot = self.fix_rot * 0.9
				camera.GetTransform().SetRot(self.fix_rot)
				if hg.Len(self.fix_rot) < 1e-4:
					self.fix_rot.x = self.fix_rot.y = self.fix_rot.z = 0
					self.flag_reseting_rotation = False
				rot = rot + self.fix_rot
			else:
				if self.mouse.Pressed(hg.MB_1):
					self.flag_reseting_rotation = True
				if not self.flag_hovering_gui:
					cam_t = camera.GetTransform()
					cam_fov = camera.GetCamera().GetFov()
					rot_fov = hg.Vec3(self.fix_rot)
					hg.FpsController(self.keyboard, self.mouse, self.fix_pos, self.fix_rot, 0, hg.time_from_sec_f(dts))
					self.fix_rot = rot_fov + (self.fix_rot - rot_fov) * cam_fov / (40/180*pi)

					#cam_pos0 = cam_t.GetPos()
					#cam_t.SetPos(cam_pos0 + (self.fps_pos - cam_pos0) * self.fps_pos_inertia)
					cam_rot0 = cam_t.GetRot()
					rot = rot + cam_rot0 + (self.fix_rot - cam_rot0) * self.fix_rot_inertia

		camera.GetTransform().SetRot(rot)

	# ============== Camera follow ==================

	def set_camera_follow_distance(self, distance):
		self.follow_distance = distance

	def update_camera_follow(self, camera: hg.Node, dts):
		self.update_target_point(dts)
		rot_mat = self.update_follow_direction(camera)
		pos = self.update_follow_translation(camera, dts)
		mat = hg.Mat4(rot_mat)
		hg.SetT(mat, pos)
		return mat

	def update_follow_direction(self, camera: hg.Node):
		v = self.target_point - camera.GetTransform().GetPos()
		rot_mat = hg.Mat3LookAt(v)
		rot = hg.ToEuler(rot_mat)
		camera.GetTransform().SetRot(rot)
		return rot_mat

	def update_follow_translation(self, camera: hg.Node, dts, init=False):
		trans = camera.GetTransform()
		camera_pos = trans.GetPos()
		aX = hg.GetX(trans.GetWorld())
		target_pos = self.target_node.GetTransform().GetPos()

		# Wall
		v = target_pos - camera_pos
		target_dir = hg.Normalize(v)
		target_dist = hg.Len(v)

		v_trans = target_dir * (target_dist - self.follow_distance) + aX * self.lateral_rot  # Déplacement latéral

		if init:
			new_position = camera_pos + v_trans
		else:
			new_position = camera_pos + v_trans * self.follow_inertia * 60 * dts
		trans.SetPos(new_position)
		self.camera_move = new_position - camera_pos
		return new_position

	# ============= Camera tracking =============================

	def set_camera_tracking_target_distance(self, distance):
		self.views["back"]["position"].z = -distance / 2
		self.views["back"]["position"].y = distance / 10

		self.views["front"]["position"].z = distance
		self.views["front"]["position"].y = distance / 10

		self.views["left"]["position"].x = distance
		self.views["left"]["position"].y = distance / 10

		self.views["right"]["position"].x = -distance
		self.views["right"]["position"].y = distance / 10

		self.views["top"]["position"].y = distance

	def update_camera_tracking(self, camera: hg.Node, dts, noise_level=0):
		self.update_target_point(dts)
		rot_mat = self.update_track_direction(camera, dts, noise_level)
		pos = self.update_track_translation(camera)
		mat = hg.Mat4(rot_mat)
		hg.SetT(mat, pos)

		return mat

	def update_track_direction(self, camera: hg.Node, dts, noise_level):
		# v = target_point - camera.GetTransform().GetPos()
		f = radians(noise_level)
		rot = hg.ToEuler(self.target_matrix)
		rot += hg.Vec3(self.noise_x.temporal_Perlin_noise(dts) * f, self.noise_y.temporal_Perlin_noise(dts) * f, self.noise_z.temporal_Perlin_noise(dts) * f)
		rot_mat = hg.RotationMat3(rot)
		rot_mat = rot_mat * self.track_orientation
		rot = hg.ToEuler(rot_mat)
		camera.GetTransform().SetRot(rot)
		return rot_mat  # hg.Mat3LookAt(v, target_matrix.GetY()))

	def update_track_translation(self, camera: hg.Node, init=False):
		trans = camera.GetTransform()
		camera_pos = trans.GetPos()
		new_position = self.target_point + hg.GetX(self.target_matrix) * self.track_position.x + hg.GetY(self.target_matrix) * self.track_position.y + hg.GetZ(self.target_matrix) * self.track_position.z
		trans.SetPos(new_position)
		self.camera_move = new_position - camera_pos
		return new_position

	# Views id: "back", "front", "left", "right", "top":
	def set_track_view(self, view_id):
		parameters = self.views[view_id]
		self.current_view = view_id
		self.track_position = parameters["position"]
		self.track_orientation = parameters["orientation"]
		self.pos_inertia = parameters["pos_inertia"]
		self.rot_inertia = parameters["rot_inertia"]

	# ======================== Satellite view ========================================

	def setup_satellite_camera(self, camera: hg.Node):
		camera.GetCamera().SetIsOrthographic(True)
		camera.GetCamera().SetSize(self.satellite_view_size)
		camera.GetTransform().SetRot(hg.Vec3(radians(90), 0, 0))

	def update_satellite_camera(self, camera, dts, init=False):
		if not init: self.update_target_point(dts)
		pos = hg.Vec3(self.target_point.x, self.target_point.y + camera.GetCamera().GetSize() * self.satellite_view_ratio, self.target_point.z)
		camera.GetTransform().SetPos(pos)
		# camera.GetTransform().SetPos(hg.Vec3(self.target_point.x, 1000, self.target_point.z))
		cam = camera.GetCamera()
		size = cam.GetSize()
		cam.SetSize(size + (self.satellite_view_size - size) * self.satellite_view_size_inertia)
		cam.SetZNear(1)
		cam.SetZFar(pos.y)

	def increment_satellite_view_size(self):
		self.satellite_view_size *= 1.01

	def decrement_satellite_view_size(self):
		self.satellite_view_size *= 0.99  # max(10, satellite_view_size / 1.01)

	# ======================== fps camera ========================================

	def update_hovering_ImGui(self):
		self.flag_hovering_gui = False
		if hg.ImGuiWantCaptureMouse() and hg.ReadMouse().Button(hg.MB_0):
			self.flag_hovering_gui = True
		if self.flag_hovering_gui and not hg.ReadMouse().Button(hg.MB_0):
			self.flag_hovering_gui = False

	def update_fps_camera(self, camera, dts):
		if not self.flag_hovering_gui:
			cam_t = camera.GetTransform()
			cam_fov = camera.GetCamera().GetFov()
			speed = 1
			if self.keyboard.Down(hg.K_LShift):
				speed = 100
			elif self.keyboard.Down(hg.K_LCtrl):
				speed = 1000
			elif self.keyboard.Down(hg.K_RCtrl):
				speed = 50000
			fps_rot_fov = hg.Vec3(self.fps_rot)
			hg.FpsController(self.keyboard, self.mouse, self.fps_pos, self.fps_rot, speed, hg.time_from_sec_f(dts))
			self.fps_rot = fps_rot_fov + (self.fps_rot - fps_rot_fov) * cam_fov / (40 / 180 * pi)

			cam_pos0 = cam_t.GetPos()
			cam_t.SetPos(cam_pos0 + (self.fps_pos - cam_pos0) * self.fps_pos_inertia)
			cam_rot0 = cam_t.GetRot()
			cam_t.SetRot(cam_rot0 + (self.fps_rot - cam_rot0) * self.fps_rot_inertia)

			if self.keyboard.Pressed(hg.K_Return):
				print("pos,rot,fov = hg.Vec3(%.3f,%.3f,%.3f),hg.Vec3(%.3f,%.3f,%.3f),%.3f" % (self.fps_pos.x, self.fps_pos.y, self.fps_pos.z, self.fps_rot.x, self.fps_rot.y, self.fps_rot.z, cam_fov))

	# ======================== Cinematic camera ========================================

	def set_keyframes(self, keyframes):
		self.keyframes = keyframes

	def update_cinematic_camera(self, camera, dts):
		if len(self.keyframes) > 0:
			cam_t = camera.GetTransform()
			# Get current tween:
			t_total = 0
			current_tween = None
			for tween in self.keyframes:
				if t_total <= self.cinematic_timer < t_total + tween["duration"]:
					current_tween = tween
					break
				else:
					t_total += tween["duration"]
			# Get t in tween
			if current_tween is None:
				# Cinematic loop
				self.cinematic_timer = 0
				t_total = 0
				current_tween = self.keyframes[0]
			t = (self.cinematic_timer - t_total) / current_tween["duration"]
			pos = current_tween["pos_start"] * (1 - t) + current_tween["pos_end"] * t
			rot = current_tween["rot_start"] * (1 - t) + current_tween["rot_end"] * t
			fov = current_tween["fov_start"] * (1 - t) + current_tween["fov_end"] * t
			cam_t.SetPos(pos)
			cam_t.SetRot(rot)
			camera.GetCamera().SetFov(fov)
			self.cinematic_timer += dts

	# ======================== Tactical camera ========================================

	def set_tactical_camera_distance(self, distance):
		self.tactical_subject_distance = distance

	def set_tactical_min_altitude(self, alt):
		self.minimal_tactical_altitude = alt

	def compute_tactical_view_center(self, camera_pos):
		v = hg.Normalize(self.subject_node.GetTransform().GetPos() - camera_pos)
		if self.target_node is not None:
			v_t = hg.Normalize(self.target_node.GetTransform().GetPos() - camera_pos)
			v += v_t
			v *= 0.5
			v = hg.Normalize(v)
		if self.projectil_node is not None:
			v_p = hg.Normalize(self.projectil_node.get_parent_node().GetTransform().GetPos() - camera_pos)
			v += v_p
			v *= 0.5
			v = hg.Normalize(v)
		return v + camera_pos

	def set_target_node(self, target_node: hg.Node):
		self.target_node = target_node

	def set_projectil_node(self, projectil_node: Destroyable_Machine):
		self.projectil_node = projectil_node

	def compute_camera_tactical_displacement(self, camera, cam_pos, target_node):
		subject_pos = self.subject_node.GetTransform().GetPos()
		dir_subject = hg.Normalize(subject_pos - cam_pos)
		cam_fov = camera.GetCamera().GetFov()
		target_pos = target_node.GetTransform().GetPos()
		dir_target = hg.Normalize(target_pos - cam_pos)
		angle = abs(acos(hg.Dot(dir_subject, dir_target)))
		if angle > cam_fov:
			# print("angle " + str(angle / pi * 180) + " FOV " + str(cam_fov / pi * 180))
			angle_diff = angle - cam_fov
			nrm = hg.Normalize(hg.Cross(dir_target, dir_subject))
			new_dir_subject = MathsSupp.rotate_vector(dir_subject, nrm, -angle_diff)
			cam_subject_dist = hg.Len(subject_pos - cam_pos)
			p = cam_pos + new_dir_subject * cam_subject_dist
			v = subject_pos - p
			return v
		return None

	def update_tactical_camera(self, camera, dts, init=False):

		# update camera position:
		cam_pos = camera.GetTransform().GetPos()
		subject_pos = self.subject_node.GetTransform().GetPos()
		dir_subject = hg.Normalize(subject_pos - cam_pos)
		v = (subject_pos - dir_subject * self.tactical_subject_distance) - cam_pos
		if init:
			cam_pos += v
		else:
			cam_pos += v  # * self.tactical_pos_inertia

		if self.target_node is not None:
			v = self.compute_camera_tactical_displacement(camera, cam_pos, self.target_node)
			if v is not None:
				cam_pos += v

		"""
		if self.projectil_node is not None:
			if self.projectil_node.activated:
				v = self.compute_camera_tactical_displacement(camera, cam_pos, self.projectil_node.get_parent_node())
				if v is not None:
					cam_pos += v
			else:
				self.set_projectil_node(None)
		"""

		t_alt, t_nrm = get_terrain_altitude(cam_pos)
		if cam_pos.y < t_alt + self.minimal_tactical_altitude:
			cam_pos.y = t_alt + self.minimal_tactical_altitude

		camera.GetTransform().SetPos(cam_pos)

		tvc = self.compute_tactical_view_center(cam_pos)
		# Update tactical view point
		v = tvc - self.target_point
		self.target_point += v

		dir = hg.Normalize(self.target_point - cam_pos)
		cam_rot = hg.ToEuler(hg.Mat3LookAt(dir, hg.Vec3.Up))
		camera.GetTransform().SetRot(cam_rot)



class Sprite:
	tex0_program = None
	spr_render_state = None
	spr_model = None
	vs_pos_tex0_decl = None

	@classmethod
	def init_system(cls):
		cls.tex0_program = hg.LoadProgramFromAssets("shaders/sprite.vsb", "shaders/sprite.fsb")

		cls.vs_pos_tex0_decl = hg.VertexLayout()
		cls.vs_pos_tex0_decl.Begin()
		cls.vs_pos_tex0_decl.Add(hg.A_Position, 3, hg.AT_Float)
		cls.vs_pos_tex0_decl.Add(hg.A_TexCoord0, 3, hg.AT_Float)
		cls.vs_pos_tex0_decl.End()
		cls.spr_model = hg.CreatePlaneModel(cls.vs_pos_tex0_decl, 1, 1, 1, 1)

		cls.spr_render_state = hg.ComputeRenderState(hg.BM_Alpha, hg.DT_Disabled, hg.FC_Disabled)

		cls.vr_size = None
		cls.vr_distance = 1


	@classmethod
	def setup_matrix_sprites2D(cls, vid, resolution: hg.Vec2):
		vs = hg.ComputeOrthographicViewState(hg.TranslationMat4(hg.Vec3(resolution.x / 2, resolution.y / 2, 0)), resolution.y, 0.1, 100, hg.Vec2(resolution.x / resolution.y, 1))
		hg.SetViewTransform(vid, vs.view, vs.proj)

	def __init__(self, w, h, texture_path):
		self.width = w
		self.height = h
		self.texture_path = texture_path
		self.texture = hg.LoadTextureFromAssets(self.texture_path, 0)[0]
		self.texture_uniform = hg.MakeUniformSetTexture("s_tex", self.texture, 0)
		self.color = hg.Color(1, 1, 1, 1)
		self.uniform_set_value_list = hg.UniformSetValueList()
		self.uniform_set_texture_list = hg.UniformSetTextureList()
		self.uniform_set_texture_list.push_back(self.texture_uniform)
		self.color_set_value = hg.MakeUniformSetValue("color", hg.Vec4(self.color.r, self.color.g, self.color.b, self.color.a))
		self.uv_scale = hg.Vec2(1, 1)
		self.uv_scale_set_value = hg.MakeUniformSetValue("uv_scale", hg.Vec4(self.uv_scale.x, self.uv_scale.y, 0, 0))
		self.position = hg.Vec3(0, 0, 2)
		self.scale = hg.Vec3(self.width, 1, self.height)
		self.rotation = hg.Vec3(0, 0, 0)
		self.size = 1

	def compute_matrix(self):
		return hg.TransformationMat4(self.position, self.rotation) * hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(90), 0, 0), self.scale * self.size)

	def set_position(self, x, y):
		self.position.x = x
		self.position.y = y

	def set_uv_scale(self, uv_scale: hg.Vec2):
		self.uv_scale = uv_scale
		self.uv_scale_set_value = hg.MakeUniformSetValue("uv_scale", hg.Vec4(self.uv_scale.x, self.uv_scale.y, 0, 0))

	def set_size(self, size):
		self.size = size

	def set_color(self, color: hg.Color):
		self.color = color
		self.color_set_value = hg.MakeUniformSetValue("color", hg.Vec4(self.color.r, self.color.g, self.color.b, self.color.a))

	def draw(self, v_id):
		self.uniform_set_value_list.clear()
		self.uniform_set_value_list.push_back(self.color_set_value)
		self.uniform_set_value_list.push_back(self.uv_scale_set_value)
		matrix = self.compute_matrix()
		hg.DrawModel(v_id, Sprite.spr_model, Sprite.tex0_program, self.uniform_set_value_list, self.uniform_set_texture_list, matrix, Sprite.spr_render_state)

	def draw_vr(self, v_id, vr_matrix, resolution, vr_hud):
		pos_vr = hg.Vec3((self.position.x / resolution.x - 0.5) * vr_hud.x, (self.position.y / resolution.y - 0.5) * vr_hud.y, vr_hud.z)
		scale_vr = hg.Vec3(self.scale.x / resolution.x * vr_hud.x, 1, self.scale.z / resolution.y * vr_hud.y)
		matrix = vr_matrix * hg.TransformationMat4(pos_vr, self.rotation) * hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(90), 0, 0), scale_vr * self.size)

		self.uniform_set_value_list.clear()
		self.uniform_set_value_list.push_back(self.color_set_value)
		self.uniform_set_value_list.push_back(self.uv_scale_set_value)
		hg.DrawModel(v_id, Sprite.spr_model, Sprite.tex0_program, self.uniform_set_value_list, self.uniform_set_texture_list, matrix, Sprite.spr_render_state)
class PostProcess:
	def __init__(self, resolution, antialiasing = 4, flag_vr=False):
		# Setup frame buffers

		self.render_program = hg.LoadProgramFromAssets("shaders/copy")

		self.flag_vr = flag_vr

		# Render frame buffer
		if flag_vr:
			self.quad_frameBuffer_left = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_postprocess_left")  # hg.OpenVRCreateEyeFrameBuffer(hg.OVRAA_MSAA4x)
			self.quad_frameBuffer_right = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_postprocess_right")  # hg.OpenVRCreateEyeFrameBuffer(hg.OVRAA_MSAA4x)
		else:
			self.quad_frameBuffer = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_postprocess")

		# Setup 2D rendering
		self.quad_mdl = hg.VertexLayout()
		self.quad_mdl.Begin()
		self.quad_mdl.Add(hg.A_Position, 3, hg.AT_Float)
		self.quad_mdl.Add(hg.A_TexCoord0, 3, hg.AT_Float)
		self.quad_mdl.End()

		self.quad_model = hg.CreatePlaneModel(self.quad_mdl, 1, 1, 1, 1)
		self.quad_uniform_set_value_list = hg.UniformSetValueList()
		self.quad_uniform_set_texture_list = hg.UniformSetTextureList()
		self.quad_render_state = hg.ComputeRenderState(hg.BM_Opaque, hg.DT_Disabled, hg.FC_Disabled)
		self.quad_matrix = hg.TransformationMat4(hg.Vec3(0, 0, 1), hg.Vec3(hg.Deg(90), hg.Deg(0), hg.Deg(0)), hg.Vec3(resolution.x, 1, resolution.y))

		self.post_process_program = hg.LoadProgramFromAssets("shaders/post_process")

		self.color = hg.Vec4(1, 1, 1, 1)
		self.uv_scale = hg.Vec4(1, 1, 0, 0)
		self.fade_t = 0
		self.fade_f = 0
		self.fade_duration = 1
		self.fade_direction = 1
		self.fade_running = False

		self.quad_uniform_set_value_list.clear()
		self.quad_uniform_set_value_list.push_back(hg.MakeUniformSetValue("uv_scale", self.uv_scale))
		self.quad_uniform_set_value_list.push_back(hg.MakeUniformSetValue("color", self.color))

	def setup_fading(self, duration, direction):
		self.fade_duration = duration
		self.fade_direction = direction
		self.fade_running = True
		self.fade_t = 0

	def update_fading(self, dts):
		if self.fade_running:
			if self.fade_t >= self.fade_duration: self.fade_running = False
			self.fade_t += dts
			self.fade_f = min(1, self.fade_t / self.fade_duration)
			if self.fade_direction < 0: self.fade_f = 1 - self.fade_f
			self.quad_uniform_set_value_list.clear()
			c = self.color * self.fade_f
			c.w = 1
			self.quad_uniform_set_value_list.push_back(hg.MakeUniformSetValue("color", c))
			self.quad_uniform_set_value_list.push_back(hg.MakeUniformSetValue("uv_scale", self.uv_scale))

	def display(self, view_id, resources, resolution, custom_texture=None):
		hg.SetViewRect(view_id, 0, 0, int(resolution.x), int(resolution.y))
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		hg.SetViewFrameBuffer(view_id, hg.InvalidFrameBufferHandle)
		vs = hg.ComputeOrthographicViewState(hg.TranslationMat4(hg.Vec3(0, 0, 0)), resolution.y, 0.1, 100, hg.ComputeAspectRatioX(resolution.x, resolution.y))
		hg.SetViewTransform(view_id, vs.view, vs.proj)
		self.quad_uniform_set_texture_list.clear()
		if custom_texture is None:
			self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.GetColorTexture(self.quad_frameBuffer), 0))
		else:
			self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", custom_texture, 0))
		hg.DrawModel(view_id, self.quad_model, self.post_process_program, self.quad_uniform_set_value_list, self.quad_uniform_set_texture_list, self.quad_matrix, self.quad_render_state)
		return view_id + 1

	def display_vr(self, view_id, vr_state: hg.OpenVRState, vs_left: hg.ViewState, vs_right: hg.ViewState, output_left_fb: hg.OpenVREyeFrameBuffer, output_right_fb: hg.OpenVREyeFrameBuffer, resources):

		focal_distance_left = hg.ExtractZoomFactorFromProjectionMatrix(vs_left.proj)
		focal_distance_right = hg.ExtractZoomFactorFromProjectionMatrix(vs_right.proj)

		z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_left.proj)
		z_ratio = (z_near + 0.01) / focal_distance_left
		hg.SetViewFrameBuffer(view_id, output_left_fb.GetHandle())
		hg.SetViewRect(view_id, 0, 0, int(vr_state.width), int(vr_state.height))
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		hg.SetViewTransform(view_id, hg.InverseFast(vr_state.left.offset), vs_left.proj)
		matrx = vr_state.left.offset * hg.TransformationMat4(hg.Vec3(0, 0, focal_distance_left * z_ratio), hg.Vec3(hg.Deg(90), hg.Deg(0), hg.Deg(0)), hg.Vec3(2, 2, 2) * z_ratio)
		self.quad_uniform_set_texture_list.clear()
		# self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.OpenVRGetColorTexture(self.quad_frameBuffer_left), 0))
		self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.GetColorTexture(self.quad_frameBuffer_left), 0))
		hg.DrawModel(view_id, self.quad_model, self.post_process_program, self.quad_uniform_set_value_list, self.quad_uniform_set_texture_list, matrx, self.quad_render_state)
		view_id += 1

		z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_right.proj)
		z_ratio = (z_near + 0.01) / focal_distance_right
		hg.SetViewFrameBuffer(view_id, output_right_fb.GetHandle())
		hg.SetViewRect(view_id, 0, 0, int(vr_state.width), int(vr_state.height))
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		hg.SetViewTransform(view_id, hg.InverseFast(vr_state.right.offset), vs_right.proj)
		matrx = vr_state.right.offset * hg.TransformationMat4(hg.Vec3(0, 0, focal_distance_right * z_ratio), hg.Vec3(hg.Deg(90), hg.Deg(0), hg.Deg(0)), hg.Vec3(2, 2, 2) * z_ratio)
		self.quad_uniform_set_texture_list.clear()
		# self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.OpenVRGetColorTexture(self.quad_frameBuffer_left), 0))
		self.quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.GetColorTexture(self.quad_frameBuffer_right), 0))
		hg.DrawModel(view_id, self.quad_model, self.post_process_program, self.quad_uniform_set_value_list, self.quad_uniform_set_texture_list, matrx, self.quad_render_state)
		return view_id + 1
class HUD:

	@classmethod
	def init(cls, resolution: hg.Vec2):

		cls.color_inactive = hg.Color(0.2, 0.2, 0.2, 0.5)
		cls.color_wait_connect = hg.Color(1, 0.8, 0.8, 1)
		cls.color_connected = hg.Color(0.3, 0.3, 0.3, 1)

	# Texts:
	@staticmethod
	def hud_convert_coords(x, y, resolution):
		ratio = resolution.x / resolution.y
		return (x - resolution.x / 2) / (resolution.x / 2) * ratio, (y - resolution.y / 2) / (resolution.y / 2)


class HUD_Radar:

	spr_radar = None
	spr_radar_light = None
	spr_radar_box = None
	aircrafts_plots = None
	missiles_plots = None
	ships_plots = None
	missile_launchers_plots = None
	dir_plots = None
	spr_noise = None

	@classmethod
	def init(cls, resolution:hg.Vec2):
		cls.spr_radar = Sprite(530, 530, "sprites/radar.png")
		cls.spr_radar_light = Sprite(530, 530, "sprites/radar_light.png")
		cls.spr_radar_box = Sprite(530, 530, "sprites/radar_box.png")
		cls.aircrafts_plots = []
		cls.missiles_plots = []
		cls.ships_plots = []
		cls.missile_launchers_plots = []
		cls.dir_plot = Sprite(32, 32, "sprites/plot_dir.png")
		cls.spr_noise = Sprite(256, 256, "sprites/noise.png")
		rs = (200 / 1600 * resolution.x) / cls.spr_radar.width
		cls.spr_radar.set_size(rs)
		cls.spr_radar_light.set_size(rs)
		cls.spr_radar_box.set_size(rs)
		cls.spr_noise.set_size((200 / 1600 * resolution.x) / cls.spr_noise.width)

	@classmethod
	def setup_plots(cls, resolution, num_aircrafts, num_missiles, num_ships, num_missile_launchers):
		cls.aircrafts_plots = []
		cls.missiles_plots = []
		cls.ships_plots = []
		cls.missile_launchers_plots = []
		for i in range(num_aircrafts):
			cls.aircrafts_plots.append(Sprite(40, 40, "sprites/plot.png"))
		for i in range(num_missiles):
			cls.missiles_plots.append(Sprite(40, 40, "sprites/plot_missile.png"))
		for i in range(num_ships):
			cls.ships_plots.append(Sprite(40, 40, "sprites/plot_ship.png"))
		for i in range(num_missile_launchers):
			cls.missile_launchers_plots.append(Sprite(40, 40, "sprites/plot_missile_launcher.png"))

	@classmethod
	def update(cls, Main, machine:Destroyable_Machine, targets):
		t = hg.time_to_sec_f(hg.GetClock())
		rx, ry = 150 / 1600 * Main.resolution.x, 150 / 900 * Main.resolution.y
		rm = 6 / 1600
		rs = cls.spr_radar.size

		radar_scale = 4000
		plot_size = 12 / 1600 * Main.resolution.x

		cls.spr_radar.set_position(rx, ry)
		cls.spr_radar.set_color(hg.Color(1, 1, 1, 1))
		Main.sprites_display_list.append(cls.spr_radar)

		mat, pos, rot, aX, aY, aZ = machine.decompose_matrix()
		aZ.y = 0
		aZ = hg.Normalize(aZ)
		if aY.y < 0:
			aY = hg.Vec3(0, -1, 0)
		else:
			aY = hg.Vec3(0, 1, 0)
		aX = hg.Cross(aY, aZ)
		matrot = hg.Mat3()
		hg.SetAxises(matrot, aX, aY, aZ)
		mat_trans = hg.InverseFast(hg.TransformationMat4(hg.GetT(mat), matrot))

		i_missile = 0
		i_ship = 0
		i_aircraft = 0
		i_missile_launcher = 0
		td = machine.get_device("TargettingDevice")

		for target in targets:
			if not target.wreck and target.activated:
				t_mat, t_pos, t_rot, aX, aY, aZ = target.decompose_matrix()
				aZ.y = 0
				aZ = hg.Normalize(aZ)
				aY = hg.Vec3(0, 1, 0)
				aX = hg.Cross(aY, aZ)
				matrot = hg.Mat3()
				hg.SetAxises(matrot, aX, aY, aZ)
				t_mat_trans = mat_trans * hg.TransformationMat4(t_pos, matrot)
				pos = hg.GetT(t_mat_trans)
				v2D = hg.Vec2(pos.x, pos.z) / radar_scale * rs / 2
				if abs(v2D.x) < rs / 2 - rm and abs(v2D.y) < rs / 2 - rm:

					if target.type == Destroyable_Machine.TYPE_MISSILE:
						plot = cls.missiles_plots[i_missile]
						i_missile += 1
					elif target.type == Destroyable_Machine.TYPE_AIRCRAFT:
						plot = cls.aircrafts_plots[i_aircraft]
						i_aircraft += 1
					elif target.type == Destroyable_Machine.TYPE_SHIP:
						plot = cls.ships_plots[i_ship]
						i_ship += 1
					elif target.type == Destroyable_Machine.TYPE_MISSILE_LAUNCHER:
						plot = cls.missile_launchers_plots[i_missile_launcher]
						i_missile_launcher += 1
					t_mat_rot = hg.GetRotationMatrix(t_mat_trans)
					a = 0.5 + 0.5 * abs(sin(t * uniform(1, 500)))
				else:
					if target.type == Destroyable_Machine.TYPE_MISSILE: continue
					dir = hg.Normalize(v2D)
					v2D = dir * (rs / 2 - rm)
					plot = cls.dir_plot
					aZ = hg.Vec3(dir.x, 0, dir.y)
					aX = hg.Cross(hg.Vec3.Up, aZ)
					t_mat_rot = hg.Mat3(aX, hg.Vec3.Up, aZ)
					a = 0.5 + 0.5 * abs(sin(t * uniform(1, 500)))

				v2D *= Main.resolution.y / 2
				cx, cy = rx + v2D.x, ry + v2D.y

				if td is not None and target == td.get_target():
					c = hg.Color(0.85, 1., 0.25, a)
				elif target.nationality == machine.nationality:
					c = hg.Color(0.25, 1., 0.25, a)
				else:
					c = hg.Color(1., 0.5, 0.5, a)

				rot = hg.ToEuler(t_mat_rot)
				plot.set_position(cx, cy)
				plot.rotation.z = -rot.y
				plot.set_size(plot_size / plot.width)
				plot.set_color(c)
				Main.sprites_display_list.append(plot)

		cls.spr_noise.set_position(rx, ry)
		cls.spr_noise.set_color(hg.Color(1, 1, 1, max(pow(1 - machine.health_level, 2), 0.05)))
		cls.spr_noise.set_uv_scale(hg.Vec2((0.75 + 0.25 * sin(t * 538)) - (0.25 + 0.25 * sin(t * 103)), (0.75 + 0.25 * cos(t * 120)) - ((0.65 + 0.35 * sin(t * 83)))))
		Main.sprites_display_list.append(cls.spr_noise)

		cls.spr_radar_light.set_position(rx, ry)
		cls.spr_radar_light.set_color(hg.Color(1, 1, 1, 0.3))
		Main.sprites_display_list.append(cls.spr_radar_light)

		cls.spr_radar_box.set_position(rx, ry)
		cls.spr_radar_box.set_color(hg.Color(1, 1, 1, 1))
		Main.sprites_display_list.append(cls.spr_radar_box)


class HUD_MachineGun:
	spr_machine_gun_sight = None

	@classmethod
	def init(cls, resolution:hg.Vec2):
		cls.spr_machine_gun_sight = Sprite(64, 64, "sprites/machine_gun_sight.png")
		cls.spr_machine_gun_sight.set_color(hg.Color(0.5, 1, 0.5, 1))

	@classmethod
	def update(cls, main, machine):
		mat, pos, rot, aX, aY, aZ = machine.decompose_matrix()
		aZ = hg.GetZ(mat)
		aZ = hg.Normalize(aZ)
		gp = hg.Vec3(0, 0, 0)
		for gs in machine.machine_gun_slots:
			gp = gp + hg.GetT(gs.GetTransform().GetWorld())
		gp = gp / len(machine.machine_gun_slots)
		p = gp + aZ * 500
		p2D = main.get_2d_hud(p)
		if p2D is not None:
			cls.spr_machine_gun_sight.set_position(p2D.x, p2D.y)
			main.sprites_display_list.append(cls.spr_machine_gun_sight)


class HUD_MissileTarget:
	spr_target_sight = None
	spr_missile_sight = None

	@classmethod
	def init(cls, resolution: hg.Vec2):
		cls.spr_target_sight = Sprite(64, 64, "sprites/target_sight.png")
		cls.spr_missile_sight = Sprite(64, 64, "sprites/missile_sight.png")

		cls.spr_target_sight.set_size((32 / 1600 * resolution.x) / cls.spr_target_sight.width)
		cls.spr_missile_sight.set_size((32 / 1600 * resolution.x) / cls.spr_missile_sight.width)

	@classmethod
	def display_selected_target(cls, main, selected_machine):
		mat, pos, rot, aX, aY, aZ = selected_machine.decompose_matrix()
		p2D = main.get_2d_hud(pos)
		if p2D is not None:
			msg = selected_machine.name
			x = (p2D.x / main.resolution.x - 12 / 1600)
			c = hg.Color(1, 1, 0.0, 1.0)
			cls.spr_target_sight.set_position(p2D.x, p2D.y)
			cls.spr_target_sight.set_color(c)
			main.sprites_display_list.append(cls.spr_target_sight)
			Overlays.add_text2D(msg, hg.Vec2(x, (p2D.y / main.resolution.y - 24 / 900)), 0.012, c, main.hud_font)

	@classmethod
	def update(cls, main, machine):
		tps = hg.time_to_sec_f(hg.GetClock())
		td = machine.get_device("TargettingDevice")
		if td is not None:
			target = td.get_target()
			f = 1  # Main.HSL_postProcess.GetL()
			if target is not None:
				p2D = main.get_2d_hud(target.get_parent_node().GetTransform().GetPos())
				if p2D is not None:
					a_pulse = 0.5 if (sin(tps * 20) > 0) else 0.75
					if td.target_locked:
						c = hg.Color(1., 0.5, 0.5, a_pulse)
						msg = "LOCKED - " + str(int(td.target_distance))
						x = (p2D.x / main.resolution.x - 32 / 1600)
						a = a_pulse
					else:
						msg = str(int(td.target_distance))
						x = (p2D.x / main.resolution.x - 12 / 1600)
						c = hg.Color(0.5, 1, 0.5, 0.75)

					c.a *= f
					cls.spr_target_sight.set_position(p2D.x, p2D.y)
					cls.spr_target_sight.set_color(c)
					main.sprites_display_list.append(cls.spr_target_sight)

					if td.target_out_of_range:
						Overlays.add_text2D("OUT OF RANGE", hg.Vec2(p2D.x / main.resolution.x - 40 / 1600, p2D.y / main.resolution.y - 24 / 900), 0.012, hg.Color(0.2, 1, 0.2, a_pulse * f), main.hud_font)
					else:
						Overlays.add_text2D(msg, hg.Vec2(x, (p2D.y / main.resolution.y - 24 / 900)), 0.012, c, main.hud_font)

					if td.target_locking_state > 0:
						t = sin(td.target_locking_state * pi - pi / 2) * 0.5 + 0.5
						p2D = hg.Vec2(0.5, 0.5) * (1 - t) + p2D * t
						cls.spr_missile_sight.set_position(p2D.x, p2D.y)
						cls.spr_missile_sight.set_color(c)
						main.sprites_display_list.append(cls.spr_missile_sight)

				c = hg.Color(0, 1, 0, f)

				Overlays.add_text2D("Target dist: %d" % (td.target_distance), hg.Vec2(0.05, 0.91), 0.016, c, main.hud_font)
				Overlays.add_text2D("Target heading: %d" % (td.target_heading),hg.Vec2(0.05, 0.89), 0.016, c, main.hud_font)
				Overlays.add_text2D("Target alt: %d" % (td.target_altitude), hg.Vec2(0.05, 0.87), 0.016, c, main.hud_font)


class HUD_Aircraft:

	@classmethod
	def update(cls, Main, aircraft: Aircraft, targets):
		f = 1  # Main.HSL_postProcess.GetL()
		tps = hg.time_to_sec_f(hg.GetClock())
		a_pulse = 0.1 if (sin(tps * 25) > 0) else 0.9
		hs, vs = aircraft.get_world_speed()
		if Main.flag_network_mode:
			if Main.flag_client_connected:
				Overlays.add_text2D("Client connected", hg.Vec2(0.05, 0.98), 0.016, HUD.color_connected * f, Main.hud_font)
			else:
				h, p = Main.get_network()
				Overlays.add_text2D("Host: " + h + " Port: " + str(p), hg.Vec2(0.05, 0.98), 0.016, HUD.color_wait_connect * f, Main.hud_font)

		if aircraft.flag_custom_physics_mode:
			Overlays.add_text2D("Custom physics", hg.Vec2(0.05, 0.92), 0.016, hg.Color.White * f, Main.hud_font)

		Overlays.add_text2D("Health: %d" % (aircraft.health_level * 127), hg.Vec2(0.05, 0.96), 0.016, (hg.Color.White * aircraft.health_level + hg.Color.Red * (1 - aircraft.health_level)) * f, Main.hud_font)

		# Compute num bullets

		Overlays.add_text2D("Bullets: %d" % (aircraft.get_num_bullets()), hg.Vec2(0.05, 0.94), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Heading: %d" % (aircraft.get_heading()), hg.Vec2(0.49, 0.96), 0.016, hg.Color.White * f, Main.hud_font)

		iactrl = aircraft.get_device("IAControlDevice")
		if iactrl.is_activated():
			c = hg.Color.Orange
		else:
			c = HUD.color_inactive
		Overlays.add_text2D("IA Activated", hg.Vec2(0.45, 0.94), 0.015, c * f, Main.hud_font)

		# Gear HUD
		gear = aircraft.get_device("Gear")
		if gear is not None:
			color_gear = hg.Color(0.8, 1, 0.2, 1)
			gst = "DOWN"
			if gear.flag_gear_moving:
				c = color_gear * a_pulse + HUD.color_inactive * (1 - a_pulse)
			elif gear.activated:
				c = color_gear
			else:
				gst = "UP"
				c = HUD.color_inactive
			Overlays.add_text2D("GEAR", hg.Vec2(0.52, 0.94), 0.015, c * f, Main.hud_font)
		else:
			Overlays.add_text2D("No gear installed", hg.Vec2(0.52, 0.94), 0.015, HUD.color_inactive * f, Main.hud_font)

		flag_internal_physics = not aircraft.get_custom_physics_mode()

		if flag_internal_physics and aircraft.playfield_distance > Destroyable_Machine.playfield_safe_distance:
			Overlays.add_text2D("Position Out of battle field !", hg.Vec2(0.43, 0.52), 0.018, hg.Color.Red * f, Main.hud_font)
			Overlays.add_text2D("Turn back now or you'll die !", hg.Vec2(0.405, 0.48), 0.018, hg.Color.Red * a_pulse * f, Main.hud_font)


		alt = aircraft.get_altitude()
		terrain_alt = aircraft.terrain_altitude
		if alt > aircraft.max_safe_altitude and flag_internal_physics :
			c = hg.Color.Red * a_pulse + hg.Color.Yellow * (1-a_pulse)
			Overlays.add_text2D("AIR DENSITY LOW - Damaging thrust", hg.Vec2(0.8, 0.95), 0.016, hg.Color.Red * f, Main.hud_font)
		else:
			c = hg.Color.White
		Overlays.add_text2D("Altitude (ft): %d" % (alt*3.2808399),  hg.Vec2(0.8, 0.93), 0.016, c * f, Main.hud_font)
		Overlays.add_text2D("Ground (ft): %d" % (terrain_alt*3.2808399),  hg.Vec2(0.8, 0.91), 0.016, c * f, Main.hud_font)

		Overlays.add_text2D("Vertical speed (m/s): %d" % (vs), hg.Vec2(0.8, 0.89), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("horizontal speed (m/s): %d" % (hs), hg.Vec2(0.8, 0.1), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Pitch: %d" % (aircraft.get_pitch_attitude()), hg.Vec2(0.8, 0.14), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Roll: %d" % (aircraft.get_roll_attitude()), hg.Vec2(0.8, 0.12), 0.016, hg.Color.White * f, Main.hud_font)

		ls = aircraft.get_linear_speed() * 3.6

		Overlays.add_text2D("Linear speed (mph): %d" % (ls*0.62137119), hg.Vec2(0.8, 0.2), 0.016, hg.Color.White * f, Main.hud_font)

		if ls < aircraft.minimum_flight_speed and not aircraft.flag_landed:
			Overlays.add_text2D("LOW SPEED", hg.Vec2(0.47, 0.13), 0.018, hg.Color(1., 0, 0, a_pulse) * f, Main.hud_font)
		if aircraft.flag_landed:
			Overlays.add_text2D("LANDED", hg.Vec2(0.48, 0.13), 0.018, hg.Color(0.2, 1, 0.2, a_pulse) * f, Main.hud_font)

		Overlays.add_text2D("Linear acceleration (m.s2): %.2f" % (aircraft.get_linear_acceleration()), hg.Vec2(0.8, 0.02), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Engines: %d N1" % (aircraft.get_thrust_level() * 127), hg.Vec2(0.47, 0.1), 0.016, hg.Color.White * f, Main.hud_font)

		if aircraft.brake_level > 0:
			Overlays.add_text2D("Brake: %d" % (aircraft.get_brake_level() * 127), hg.Vec2(0.43, 0.046), 0.016, hg.Color(1, 0.4, 0.4) * f, Main.hud_font)

		if aircraft.flaps_level > 0:
			Overlays.add_text2D("Flaps: %d" % (aircraft.get_flaps_level() * 127), hg.Vec2(0.515, 0.046), 0.016, hg.Color(1, 0.8, 0.4) * f, Main.hud_font)

		HUD_Radar.update(Main, aircraft, targets)
		HUD_MissileTarget.update(Main, aircraft)

		if not Main.satellite_view:
			HUD_MachineGun.update(Main, aircraft)


class HUD_MissileLauncher:

	@classmethod
	def update(cls, Main, aircraft:MissileLauncherS400, targets):
		f = 1  # Main.HSL_postProcess.GetL()
		tps = hg.time_to_sec_f(hg.GetClock())
		a_pulse = 0.1 if (sin(tps * 25) > 0) else 0.9
		if Main.flag_network_mode:
			if Main.flag_client_connected:
				Overlays.add_text2D("Client connected", hg.Vec2(0.05, 0.98), 0.016, HUD.color_connected * f, Main.hud_font)
			else:
				h, p = Main.get_network()
				Overlays.add_text2D("Host: " + h + " Port: " + str(p), hg.Vec2(0.05, 0.98), 0.016, HUD.color_wait_connect * f, Main.hud_font)

		if aircraft.flag_custom_physics_mode:
			Overlays.add_text2D("Custom physics", hg.Vec2(0.05, 0.92), 0.016, hg.Color.White * f, Main.hud_font)

		Overlays.add_text2D("Health: %d" % (aircraft.health_level * 127), hg.Vec2(0.05, 0.96), 0.016, (hg.Color.White * aircraft.health_level + hg.Color.Red * (1 - aircraft.health_level)) * f, Main.hud_font)

		Overlays.add_text2D("Heading: %d" % (aircraft.get_heading()), hg.Vec2(0.49, 0.96), 0.016, hg.Color.White * f, Main.hud_font)

		iactrl = aircraft.get_device("IAControlDevice")
		if iactrl is not None:
			if iactrl.is_activated():
				c = hg.Color.Orange
			else:
				c = HUD.color_inactive
			Overlays.add_text2D("IA Activated", hg.Vec2(0.45, 0.94), 0.015, c * f, Main.hud_font)

		flag_internal_physics = not aircraft.get_custom_physics_mode()

		if flag_internal_physics and aircraft.playfield_distance > Destroyable_Machine.playfield_safe_distance:
			Overlays.add_text2D("Position Out of battle field !", hg.Vec2(0.43, 0.52), 0.018, hg.Color.Red * f, Main.hud_font)
			Overlays.add_text2D("Turn back now or you'll be destroyed !", hg.Vec2(0.405, 0.48), 0.018, hg.Color.Red * a_pulse * f, Main.hud_font)

		alt = aircraft.get_altitude()
		terrain_alt = aircraft.terrain_altitude
		c = hg.Color.White
		Overlays.add_text2D("Altitude (ft): %d" % (alt*3.2808399),  hg.Vec2(0.8, 0.93), 0.016, c * f, Main.hud_font)
		Overlays.add_text2D("Ground (ft): %d" % (terrain_alt*3.2808399),  hg.Vec2(0.8, 0.91), 0.016, c * f, Main.hud_font)

		Overlays.add_text2D("Pitch : %d" % (aircraft.get_pitch_attitude()), hg.Vec2(0.8, 0.14), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Roll : %d" % (aircraft.get_roll_attitude()), hg.Vec2(0.8, 0.12), 0.016, hg.Color.White * f, Main.hud_font)

		ls = aircraft.get_linear_speed() * 3.6

		Overlays.add_text2D("Linear speed (mph): %d" % (ls*0.62137119), hg.Vec2(0.8, 0.06), 0.016, hg.Color.White * f, Main.hud_font)

		Overlays.add_text2D("Linear acceleration (m.s2): %.2f" % (aircraft.get_linear_acceleration()), hg.Vec2(0.8, 0.02), 0.016, hg.Color.White * f, Main.hud_font)
		Overlays.add_text2D("Engines: %d N1" % (aircraft.get_thrust_level() * 127), hg.Vec2(0.47, 0.1), 0.016, hg.Color.White * f, Main.hud_font)

		if aircraft.brake_level > 0:
			Overlays.add_text2D("Brake: %d" % (aircraft.get_brake_level() * 127), hg.Vec2(0.43, 0.046), 0.016, hg.Color(1, 0.4, 0.4) * f, Main.hud_font)

		HUD_Radar.update(Main, aircraft, targets)
		HUD_MissileTarget.update(Main, aircraft)
class Animations:
	animations = []

	@staticmethod
	def interpolation_lineaire(a, b, t):
		return a * (1 - t) + b * t

	@staticmethod
	def interpolation_cosinusoidale(a, b, t):
		return Animations.interpolation_lineaire(a, b, (-cos(pi * t) + 1) / 2)

	@classmethod
	def update_animations(cls, t):
		for anim in cls.animations:
			anim.update(t)


class Animation:
	def __init__(self, t_start, delay, v_start, v_end):
		self.t_start = t_start
		self.delay = delay
		self.v_start = v_start
		self.v_end = v_end
		self.v = v_start
		self.flag_end = False

	def update(self, t):
		if t > self.t_start + self.delay:
			self.v = self.v_end
			self.flag_end = True
		elif t >= self.t_start:
			self.v = Animations.interpolation_cosinusoidale(self.v_start, self.v_end, (t - self.t_start) / self.delay)
		else:
			self.v = self.v_start
class Mission:
        def __init__(self, title, ennemies:list, allies:list, num_carriers_ennemies:int, num_carriers_allies:int, setup_players, end_test, end_phase_update):
                self.title = title
                self.ennemies = ennemies
                self.allies = allies
                self.allies_carriers = num_carriers_allies
                self.ennemies_carriers = num_carriers_ennemies
                self.setup_players_f = setup_players
                self.end_test_f = end_test
                self.update_end_phase_f = end_phase_update
                self.failed = False
                self.aborted = False

        def reset(self):
                self.failed = False
                self.aborted = False

        def setup_players(self, main):
                self.setup_players_f(main)

        def end_test(self, main):
                return self.end_test_f(main)

        def update_end_phase(self, main, dts):
                self.update_end_phase_f(main, dts)

class Missions:
        beep_ref = None
        beep_state = None
        beep_source = None

        validation_ref = None
        validation_state = None
        validation_source = None

        # animations mission:
        am_start = False
        am_anim_x_prec = None
        am_anim_a_prec = None
        am_anim_x_new = None
        am_anim_a_new = None
        am_mission_id_prec = 0
        am_t = 0

        missions = []
        mission_id = 1

        @classmethod
        def display_mission_title(cls, main, fade_lvl, dts, yof7):
                mcr = 1
                mcg = 0.6
                mcb = 0.1

                if not cls.am_start:
                        mx = 0.5
                        mc = hg.Color(mcr, mcg, mcb, 1) * fade_lvl
                        mid = cls.mission_id
                        if cls.mission_id < len(cls.missions) - 1:
                                if get_connected()[0]:
                                        if get_button_values(get_state(0))['DPAD_RIGHT']:
                                                cls.am_start = True
                                                mcpt = 1
                                                xpd = 0
                                                xns = 1     
                                elif main.keyboard.Pressed(hg.K_Right):
                                        cls.am_start = True
                                        mcpt = 1
                                        xpd = 0
                                        xns = 1
                        if cls.mission_id > 0:
                                if get_connected()[0]:
                                        if get_button_values(get_state(0))['DPAD_LEFT']:
                                                cls.am_start = True
                                                mcpt = -1
                                                xpd = 1
                                                xns = 0
                                elif main.keyboard.Pressed(hg.K_Left):
                                        cls.am_start = True
                                        mcpt = -1
                                        xpd = 1
                                        xns = 0

                        if cls.am_start:
                                cls.beep_source = hg.PlayStereo(cls.beep_ref, cls.beep_state)
                                hg.SetSourceVolume(cls.beep_source, 0.05)
                                cls.am_mission_id_prec = cls.mission_id
                                cls.mission_id += mcpt
                                cls.am_t = 0
                                cls.am_anim_x_prec = Animation(0, 0.5, 0.5, xpd)
                                cls.am_anim_a_prec = Animation(0, 0.5, 1, 0)
                                cls.am_anim_x_new = Animation(0, 0.2, xns, 0.5)
                                cls.am_anim_a_new = Animation(0, 0.2, 0, 1)
                                mid = cls.am_mission_id_prec

                else:
                        cls.am_t += dts
                        cls.am_anim_x_prec.update(cls.am_t)
                        cls.am_anim_x_new.update(cls.am_t)
                        cls.am_anim_a_prec.update(cls.am_t)
                        cls.am_anim_a_new.update(cls.am_t)

                        mx = cls.am_anim_x_new.v
                        mc = hg.Color(mcr, mcg, mcb, cls.am_anim_a_new.v) * fade_lvl

                        mxprec = cls.am_anim_x_prec.v
                        mcprec = hg.Color(mcr, mcg, mcb, cls.am_anim_a_prec.v) * fade_lvl

                        Overlays.add_text2D( Missions.missions[cls.am_mission_id_prec].title, hg.Vec2(mxprec, 671 / 900 + yof7), 0.02, mcprec, main.hud_font, hg.DTHA_Center)
                        mid = cls.mission_id

                        if cls.am_anim_a_new.flag_end and cls.am_anim_x_new.flag_end and cls.am_anim_x_prec.flag_end and cls.am_anim_a_prec:
                                cls.am_start = False

                Overlays.add_text2D(cls.missions[mid].title, hg.Vec2(mx, 671 / 900 + yof7), 0.02, mc, main.hud_font, hg.DTHA_Center)
                Overlays.add_text2D( "<- choose your mission ->", hg.Vec2(0.5, 701 / 900 + yof7), 0.012, hg.Color(1, 0.9, 0.3, fade_lvl * 0.8), main.hud_font, hg.DTHA_Center)

                if main.keyboard.Pressed(hg.K_Space):
                        cls.validation_source = hg.PlayStereo(cls.validation_ref, cls.validation_state)
                        hg.SetSourceVolume(cls.validation_source, 1)

        @classmethod
        def get_current_mission(cls):
                return cls.missions[cls.mission_id]

        @classmethod
        def aircrafts_starts_on_carrier(cls, aircrafts, carrier, start, y_orientation, distance, liftoff_delay=0, liftoff_offset=1):
                p, r = carrier.get_aircraft_start_point(0)
                carrier_alt = p.y
                carrier_mat = hg.TransformationMat4(carrier.get_parent_node().GetTransform().GetPos(), carrier.get_parent_node().GetTransform().GetRot())
                for i, aircraft in enumerate(aircrafts):
                        ia_ctrl = aircraft.get_device("IAControlDevice")
                        if ia_ctrl is not None:
                                ia_ctrl.IA_liftoff_delay = liftoff_delay
                        aircraft.flag_landed = True
                        liftoff_delay += liftoff_offset
                        start += distance
                        gear = aircraft.get_device("Gear")
                        if gear is not None:
                                bottom_height = gear.gear_height
                                gear.record_start_state(True)
                                gear.reset()
                        else:
                                bottom_height = aircraft.bottom_height
                        start.y = carrier_alt + bottom_height
                        mat = carrier_mat * hg.TransformationMat4(start, hg.Vec3(0, y_orientation, 0))
                        aircraft.reset_matrix(hg.GetT(mat), hg.GetR(mat))
                        aircraft.record_start_state()

        @classmethod
        def setup_aircrafts_on_carrier(cls, players, aircraft_carriers, start_time):
                na = len(players)
                nc = len(aircraft_carriers)
                na_row = na // (2 * nc)

                for i in range(nc):
                        n0 = na_row * (2 * i)
                        n1 = na_row * (2 * i + 1)
                        n2 = na_row * (2 * i + 2)
                        if na - n2 <= (na_row+1):
                                n2 = na
                        cls.aircrafts_starts_on_carrier(players[n0:n1], aircraft_carriers[i], hg.Vec3(10, 19.5, 80), 0, hg.Vec3(0, 0, -18), start_time + na_row * 2 * i, 2)
                        cls.aircrafts_starts_on_carrier(players[n1:n2], aircraft_carriers[i], hg.Vec3(-10, 19.5, 62), 0, hg.Vec3(0, 0, -18), start_time + 1 + na_row * 2 * i, 2)

        @classmethod
        def aircrafts_starts_in_sky(cls, aircrafts, center: hg.Vec3, range: hg.Vec3, y_orientations_range: hg.Vec2, speed_range: hg.Vec2):
                for i, ac in enumerate(aircrafts):
                        # ac.reset_matrix(hg.Vec3(uniform(center.x-range.x/2, center.x+range.x/2), uniform(center.y-range.y/2, center.y+range.y/2), uniform(center.z-range.z/2, center.z+range.z/2)), hg.Vec3(0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0))

                        ac.reset_matrix(hg.Vec3(uniform(center.x - range.x / 2, center.x + range.x / 2), uniform(center.y - range.y / 2, center.y + range.y / 2), uniform(center.z - range.z / 2, center.z + range.z / 2)), hg.Vec3(0, radians(uniform(y_orientations_range.x, y_orientations_range.y)), 0))
                        gear = ac.get_device("Gear")
                        if gear is not None:
                                gear.record_start_state(False)
                                gear.reset()
                        ac.set_linear_speed(uniform(speed_range.x, speed_range.y))
                        ac.flag_landed = False
                        ac.record_start_state()

        @classmethod
        def setup_carriers(cls, carriers, start_pos, dist, y_orientation):
                for carrier in carriers:
                        carrier.reset_matrix(start_pos, hg.Vec3(0, y_orientation, 0))
                        start_pos += dist


# ============================= Training

        @classmethod
        def mission_setup_training(cls, main):

                mission = cls.get_current_mission()
                main.create_aircraft_carriers(mission.allies_carriers, mission.ennemies_carriers)
                main.create_players(mission.allies, mission.ennemies)

                cls.setup_carriers(main.aircraft_carrier_allies, hg.Vec3(0, 0, 0), hg.Vec3(500, 0, 100), 0)

                n = len(main.players_allies)
                # if n == 1:

                cls.aircrafts_starts_on_carrier(main.players_allies, main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20))

                # cls.aircrafts_starts_in_sky(main.players_allies, hg.Vec3(1000, 2000, 0), hg.Vec3(1000, 1000, 20000), hg.Vec2(-180, 180), hg.Vec2(600 / 3.6, 800 / 3.6))

                # elif n > 1:
                #       cls.aircrafts_starts_on_carrier(main.players_allies[0:n // 2], main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20))
                #       cls.aircrafts_starts_on_carrier(main.players_allies[n // 2:n], main.aircraft_carrier_allies[0], hg.Vec3(-10, 19.5, 60), 0, hg.Vec3(0, 0, -20))

                fps_start_matrix = main.aircraft_carrier_allies[0].fps_start_point.GetTransform().GetWorld()
                main.camera_fps.GetTransform().SetWorld(fps_start_matrix)

                lt = []
                for carrier in main.aircraft_carrier_allies:
                        lt += carrier.landing_targets
                for ac in main.players_allies:
                        ac.set_landing_targets(lt)
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.activate()

                # ------- Missile Launcher:
                main.create_missile_launchers(0, 1)

                plateform = main.scene.GetNode("platform.S400")
                ml = main.missile_launchers_ennemies[0]
                ml.set_platform(plateform)

                # --------- Views setup
                main.setup_views_carousel(True)
                main.set_view_carousel("Aircraft_ally_" + str(main.num_players_allies))
                main.set_track_view("back")

                main.user_aircraft = main.get_player_from_caroursel_id(main.views_carousel[main.views_carousel_ptr])
                main.user_aircraft.set_focus()

                """
                ia = main.user_aircraft.get_device("IAControlDevice")
                if ia is not None:
                        ia.set_IA_landing_target(main.aircraft_carrier_allies[0].landing_targets[1])
                        ia.deactivate()
                uctrl = main.user_aircraft.get_device("UserControlDevice")
                if uctrl is not None:
                        uctrl.activate()
                """
                # Deactivate IA:
                for i, ac in enumerate(main.players_allies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()

                for i, ac in enumerate(main.players_ennemies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()

                uctrl = main.user_aircraft.get_device("UserControlDevice")
                if uctrl is not None:
                        uctrl.activate()

                main.init_playground()

        # if main.num_players_allies < 4:
        # main.user_aircraft.set_thrust_level(1)
        # main.user_aircraft.activate_post_combustion()

        @classmethod
        def mission_training_end_test(cls, main):
                mission = cls.get_current_mission()
                allies_wreck = 0
                for ally in main.players_allies:
                        if ally.wreck: allies_wreck += 1

                if main.num_players_allies == allies_wreck:
                        mission.failed = True
                        print("MISSION FAILED !")
                        return True

                return False

        @classmethod
        def mission_training_end_phase_update(cls, main, dts):
                mission = cls.get_current_mission()
                if mission.failed:
                        msg_title = "YOU DIED !"
                        msg_debriefing = " You need more lessons commander..."
                elif mission.aborted:
                        msg_title = "YOU QUIT !"
                        msg_debriefing = "Return to battle ?"
                else:
                        msg_title = "SUCCESSFUL !"
                        msg_debriefing = "Mission Accomplished !"
                Overlays.add_text2D(msg_title, hg.Vec2(0.5, 771 / 900 - 0.15), 0.04, hg.Color(1, 0.9, 0.3, 1), main.title_font, hg.DTHA_Center)
                Overlays.add_text2D(msg_debriefing, hg.Vec2(0.5, 701 / 900 - 0.15), 0.02, hg.Color(1, 0.9, 0.8, 1), main.hud_font, hg.DTHA_Center)

# ============================= Basic fight

        @classmethod
        def mission_setup_players(cls, main):
                mission = cls.get_current_mission()
                main.create_aircraft_carriers(mission.allies_carriers, mission.ennemies_carriers)
                main.create_players(mission.allies, mission.ennemies)

                cls.setup_carriers(main.aircraft_carrier_allies, hg.Vec3(0, 0, 0), hg.Vec3(500, 0, 100), 0)
                cls.setup_carriers(main.aircraft_carrier_ennemies, hg.Vec3(-17000, 0, 0), hg.Vec3(500, 0, -150), radians(90))

                main.init_playground()

                n = len(main.players_allies)
                if n == 1:
                        cls.aircrafts_starts_on_carrier(main.players_allies, main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20))
                elif n > 1:
                        cls.aircrafts_starts_on_carrier(main.players_allies[0:n // 2], main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20), 2, 2)
                        cls.aircrafts_starts_on_carrier(main.players_allies[n // 2:n], main.aircraft_carrier_allies[0], hg.Vec3(-10, 19.5, 60), 0, hg.Vec3(0, 0, -20), 3, 2)

                n = len(main.players_ennemies)
                if n < 3:
                        cls.aircrafts_starts_in_sky(main.players_ennemies, hg.Vec3(-5000, 1000, 0), hg.Vec3(1000, 500, 2000), hg.Vec2(-180, 180), hg.Vec2(800 / 3.6, 600 / 3.6))
                else:
                        cls.aircrafts_starts_in_sky(main.players_ennemies[0:2], hg.Vec3(-5000, 1000, 0), hg.Vec3(1000, 500, 2000), hg.Vec2(-180, 180), hg.Vec2(800 / 3.6, 600 / 3.6))
                        cls.aircrafts_starts_on_carrier(main.players_ennemies[2:n], main.aircraft_carrier_ennemies[0], hg.Vec3(-10, 19.5, 60), 0, hg.Vec3(0, 0, -20), 2, 1)

                for i, ac in enumerate(main.players_allies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.activate()

                for i, ac in enumerate(main.players_ennemies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.activate()

                main.setup_views_carousel(False)
                main.set_view_carousel("Aircraft_ally_" + str(main.num_players_allies))
                main.set_track_view("back")

                main.user_aircraft = main.get_player_from_caroursel_id(main.views_carousel[main.views_carousel_ptr])
                main.user_aircraft.set_focus()

                if main.user_aircraft is not None:
                        ia = main.user_aircraft.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()
                        uctrl = main.user_aircraft.get_device("UserControlDevice")
                        if uctrl is not None:
                                uctrl.activate()
                        if main.num_players_allies < 3:
                                main.user_aircraft.reset_thrust_level(1)
                                main.user_aircraft.activate_post_combustion()

        @classmethod
        def mission_one_against_x_end_test(cls, main):
                mission = cls.get_current_mission()
                ennemies_wreck = 0
                allies_wreck = 0
                for ennemy in main.players_ennemies:
                        if ennemy.wreck: ennemies_wreck += 1
                for ally in main.players_allies:
                        if ally.wreck: allies_wreck += 1

                if main.num_players_ennemies == ennemies_wreck:
                        mission.failed = False
                        return True
                if main.num_players_allies == allies_wreck:
                        mission.failed = True
                        return True

                return False

        @classmethod
        def mission_one_against_x_end_phase_update(cls, main, dts):
                mission = cls.get_current_mission()
                if mission.failed:
                        msg_title = "YOU DIED !"
                        msg_debriefing = " R.I.P. Commander..."
                elif mission.aborted:
                        msg_title = "YOU QUIT !"
                        msg_debriefing = "We hope you do better next time !"
                else:
                        msg_title = "SUCCESSFUL !"
                        msg_debriefing = "Mission accomplished !"
                Overlays.add_text2D(msg_title, hg.Vec2(0.5, 771 / 900 - 0.15), 0.04, hg.Color(1, 0.9, 0.3, 1), main.title_font, hg.DTHA_Center)
                Overlays.add_text2D(msg_debriefing, hg.Vec2(0.5, 701 / 900 - 0.15), 0.02, hg.Color(1, 0.9, 0.8, 1), main.hud_font, hg.DTHA_Center)

# ============================ War fight
        @classmethod
        def mission_total_war_setup_players(cls, main):

                mission = cls.get_current_mission()
                main.create_aircraft_carriers(mission.allies_carriers, mission.ennemies_carriers)
                main.create_players(mission.allies, mission.ennemies)

                cls.setup_carriers(main.aircraft_carrier_allies, hg.Vec3(0, 0, 0), hg.Vec3(300, 0, 25), 0)
                cls.setup_carriers(main.aircraft_carrier_ennemies, hg.Vec3(-20000, 0, 0), hg.Vec3(50, 0, -350), radians(90))


                cls.setup_aircrafts_on_carrier(main.players_allies, main.aircraft_carrier_allies, 0)

                n = len(main.players_ennemies)
                cls.aircrafts_starts_in_sky(main.players_ennemies[0:n // 2], hg.Vec3(-8000, 1000, 0), hg.Vec3(2000, 500, 2000), hg.Vec2(-180, -175), hg.Vec2(800 / 3.6, 600 / 3.6))

                cls.setup_aircrafts_on_carrier(main.players_ennemies[n//2:n], main.aircraft_carrier_ennemies, 60)
                
                main.init_playground()

                for i, ac in enumerate(main.players_allies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.activate()

                for i, ac in enumerate(main.players_ennemies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.activate()

                main.setup_views_carousel()
                main.set_view_carousel("Aircraft_ally_" + str(main.num_players_allies))
                main.set_track_view("back")

                main.user_aircraft = main.get_player_from_caroursel_id(main.views_carousel[main.views_carousel_ptr])
                main.user_aircraft.set_focus()

                if main.user_aircraft is not None:
                        ia = main.user_aircraft.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()
                        uctrl = main.user_aircraft.get_device("UserControlDevice")
                        if uctrl is not None:
                                uctrl.activate()
                        if main.num_players_allies < 4:
                                main.user_aircraft.reset_thrust_level(1)
                                main.user_aircraft.activate_post_combustion()

        @classmethod
        def mission_war_end_test(cls, main):
                mission = cls.get_current_mission()
                if main.keyboard.Pressed(hg.K_F6):
                        for pl in main.players_allies:
                                pl.flag_IA_start_liftoff = True
                        for pl in main.players_ennemies:
                                pl.flag_IA_start_liftoff = True
                # Pour le moment, même test de fin de mission
                return cls.mission_one_against_x_end_test(main)

        @classmethod
        def mission_war_end_phase_update(cls, main, dts):
                mission = cls.get_current_mission()
                if mission.failed:
                        msg_title = "YOU DIED !"
                        msg_debriefing = " R.I.P. Commander..."
                elif mission.aborted:
                        msg_title = "YOU QUIT !"
                        msg_debriefing = "We hope you do better next time !"
                else:
                        msg_title = "SUCCESS !"
                        msg_debriefing = "Congratulations commander !"

                Overlays.add_text2D(msg_title, hg.Vec2(0.5, 771 / 900 - 0.15), 0.04, hg.Color(1, 0.9, 0.3, 1), main.title_font, hg.DTHA_Center)
                Overlays.add_text2D(msg_debriefing, hg.Vec2(0.5, 701 / 900 - 0.15), 0.02, hg.Color(1, 0.9, 0.8, 1), main.hud_font, hg.DTHA_Center)

# ============================ Client / Server mode
        @classmethod
        def network_mode_setup(cls, main):
                mission = cls.get_current_mission()
                main.flag_network_mode = True

                file_name = "scripts/network_mission_config.json"
                file = hg.OpenText(file_name)
                if not file:
                        print("Can't open mission configuration json file : " + file_name)
                else:
                        json_script = hg.ReadString(file)
                        hg.Close(file)
                        if json_script != "":
                                script_parameters = json.loads(json_script)
                                mission.allies = script_parameters["aircrafts_allies"]
                                mission.ennemies = script_parameters["aircrafts_ennemies"]
                                mission.allies_carriers = script_parameters["aircraft_carriers_allies_count"]
                                mission.ennemies_carriers = script_parameters["aircraft_carriers_enemies_count"]
                        else:
                                print("Mission configuration json file empty : " + file_name)

                main.create_aircraft_carriers(mission.allies_carriers, mission.ennemies_carriers)
                main.create_players(mission.allies, mission.ennemies)

                cls.setup_carriers(main.aircraft_carrier_allies, hg.Vec3(0, 0, 0), hg.Vec3(500, 0, 100), 0)
                cls.setup_carriers(main.aircraft_carrier_ennemies, hg.Vec3(-17000, 0, 0), hg.Vec3(500, 0, -150), radians(90))

                main.init_playground()

                # Sets aircrafts landed on carriers:

                n = len(main.players_allies)
                if n == 1:
                        cls.aircrafts_starts_on_carrier(main.players_allies, main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20))
                elif n > 1:
                        cls.aircrafts_starts_on_carrier(main.players_allies[0:n // 2], main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20), 2, 2)
                        cls.aircrafts_starts_on_carrier(main.players_allies[n // 2:n], main.aircraft_carrier_allies[0], hg.Vec3(-10, 19.5, 60), 0, hg.Vec3(0, 0, -20), 3, 2)

                n = len(main.players_ennemies)
                if n == 1:
                        cls.aircrafts_starts_on_carrier(main.players_ennemies, main.aircraft_carrier_ennemies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20))
                elif n > 1:
                        cls.aircrafts_starts_on_carrier(main.players_ennemies[0:n // 2], main.aircraft_carrier_ennemies[0], hg.Vec3(10, 19.5, 40), 0, hg.Vec3(0, 0, -20), 2, 2)
                        cls.aircrafts_starts_on_carrier(main.players_ennemies[n // 2:n], main.aircraft_carrier_ennemies[0], hg.Vec3(-10, 19.5, 60), 0, hg.Vec3(0, 0, -20), 3, 2)

                # Deactivate IA:
                for i, ac in enumerate(main.players_allies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()

                for i, ac in enumerate(main.players_ennemies):
                        ia = ac.get_device("IAControlDevice")
                        if ia is not None:
                                ia.deactivate()

                # ------- Missile Launcher:
                main.create_missile_launchers(0, 1)

                plateform = main.scene.GetNode("platform.S400")
                ml = main.missile_launchers_ennemies[0]
                ml.set_platform(plateform)

                # --------- Views setup
                main.setup_views_carousel(True)
                main.set_view_carousel("Aircraft_ally_1")# + str(main.num_players_allies))
                main.set_track_view("back")

                main.init_playground()

                main.user_aircraft = main.get_player_from_caroursel_id(main.views_carousel[main.views_carousel_ptr])
                main.user_aircraft.set_focus()

                uctrl = main.user_aircraft.get_device("UserControlDevice")
                if uctrl is not None:
                        uctrl.activate()

                init_server(main)
                start_server()

        @classmethod
        def network_mode_end_test(cls, main):
                """
                mission = cls.get_current_mission()

                allies_wreck = 0
                for ally in main.players_allies:
                        if ally.wreck:
                                allies_wreck += 1

                if main.num_players_allies == allies_wreck:
                        mission.failed = True
                        return True
                """
                return False

        @classmethod
        def network_mode_end_phase_update(cls, main, dts):
                if main.flag_network_mode:
                        main.flag_network_mode = False
                        stop_server()
                mission = cls.get_current_mission()
                if mission.failed:
                        msg_title = "YOU DIED !"
                        msg_debriefing = " Aircraft destroyed, TAB to restart"
                elif mission.aborted:
                        msg_title = "YOU QUIT !"
                        msg_debriefing = "TAB or BACK to restart"
                else:
                        msg_title = "SUCCESSFUL !"
                        msg_debriefing = "Congratulations commander !"
                Overlays.add_text2D(msg_title, hg.Vec2(0.5, 771 / 900 - 0.15), 0.04, hg.Color(1, 0.9, 0.3, 1), main.title_font, hg.DTHA_Center)
                Overlays.add_text2D(msg_debriefing, hg.Vec2(0.5, 701 / 900 - 0.15), 0.02, hg.Color(1, 0.9, 0.8, 1), main.hud_font, hg.DTHA_Center)

        @classmethod
        # Aircrafts currently available: "F14", "F14_2", "Rafale", "Eurofighter", "F16", "TFX", "Miuss"
        def init(cls):
                cls.beep_ref = hg.LoadWAVSoundAsset("sfx/blip.wav")
                cls.beep_state = create_stereo_sound_state(hg.SR_Once)
                cls.beep_state.volume = 0.25

                cls.validation_ref = hg.LoadWAVSoundAsset("sfx/blip2.wav")
                cls.validation_state = create_stereo_sound_state(hg.SR_Once)
                cls.validation_state.volume = 0.5

                cls.missions.append(Mission("Network mode", ["Eurofighter"], ["Rafale"], 1, 1, Missions.network_mode_setup, Missions.network_mode_end_test, Missions.network_mode_end_phase_update))

                cls.missions.append(Mission("Training-Rafale", [], ["Rafale"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                cls.missions.append(Mission("Training-Eurofighter", [], ["Eurofighter"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                cls.missions.append(Mission("Training-TFX", [], ["TFX"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                cls.missions.append(Mission("Training-F16", [], ["F16"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                cls.missions.append(Mission("Training-Miuss", [], ["Miuss"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                #cls.missions.append(Mission("Training with F14", [], ["F14"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))
                #cls.missions.append(Mission("Training with F14 2", [], ["F14_2"], 0, 1, Missions.mission_setup_training, Missions.mission_training_end_test, Missions.mission_training_end_phase_update))

                cls.missions.append(Mission("Battle-1v1", ["Rafale"], ["Eurofighter"], 1, 1, Missions.mission_setup_players, Missions.mission_one_against_x_end_test, Missions.mission_one_against_x_end_phase_update))
                cls.missions.append(Mission("Battle-1v2", ["Rafale"] * 2, ["Eurofighter"], 1, 1, Missions.mission_setup_players, Missions.mission_one_against_x_end_test, Missions.mission_one_against_x_end_phase_update))
                cls.missions.append(Mission("Battle-1v3", ["Rafale"] * 1 + ["F16"] * 2, ["Eurofighter"], 1, 1, Missions.mission_setup_players, Missions.mission_one_against_x_end_test, Missions.mission_one_against_x_end_phase_update))
                cls.missions.append(Mission("Battle-1v4", ["Rafale"] * 2 + ["F16"] * 2, ["TFX", "Eurofighter"], 1, 1, Missions.mission_setup_players, Missions.mission_one_against_x_end_test, Missions.mission_one_against_x_end_phase_update))
                cls.missions.append(Mission("Battle-1v5", ["Rafale"] * 5, ["TFX", "Eurofighter", "F16"], 1, 1, Missions.mission_setup_players, Missions.mission_one_against_x_end_test, Missions.mission_one_against_x_end_phase_update))

                cls.missions.append(Mission("All Out War-5v5", ["Rafale"] * 3 + ["Eurofighter"] * 2, ["TFX"] * 2 + ["F16"] * 2 + ["Eurofighter"] * 1, 1, 1, Missions.mission_total_war_setup_players, Missions.mission_war_end_test, Missions.mission_war_end_phase_update))
                #cls.missions.append(Mission("Total War: 12 allies against 12 ennemies", ["Rafale"] * 12, ["TFX"] * 4 + ["Eurofighter"] * 4 + ["F16"] * 4 + ["Eurofighter"] * 4, 2, 2, Missions.mission_total_war_setup_players, Missions.mission_war_end_test, Missions.mission_war_end_phase_update))
                #cls.missions.append(Mission("Crash test: 60 allies against 60 ennemies", ["Rafale"] * 30 + ["Eurofighter"] * 30, ["TFX"] * 30 + ["Eurofighter"] * 20 + ["F16"] * 10, 5, 5, Missions.mission_total_war_setup_players, Missions.mission_war_end_test, Missions.mission_war_end_phase_update))
class VRState:

	def __init__(self):

		body_mtx = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(0, 0, 0))
		vr_state = hg.OpenVRGetState(body_mtx, 1, 1000)
		# Local head initial matrix:
		self.head_matrix = vr_state.head
		self.initial_head_matrix = hg.TransformationMat4(hg.GetT(self.head_matrix), hg.GetRotationMatrix(self.head_matrix))  # hg.InverseFast(vr_state.body) * vr_state.head
		# Local eyes offsets:
		self.eye_left_offset = vr_state.left.offset
		self.eye_right_offset = vr_state.right.offset
		# Fov
		vs_left, vs_right = hg.OpenVRStateToViewState(vr_state)
		self.fov_left = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_left.proj))
		self.fov_right = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_right.proj))
		self.resolution = hg.Vec2(vr_state.width, vr_state.height)
		self.ratio = hg.Vec2(self.resolution.x / self.resolution.y, 1)

	def update_initial_head_matrix(self):
		self.initial_head_matrix = hg.TransformationMat4(hg.GetT(self.head_matrix), hg.GetRotationMatrix(self.head_matrix))  # hg.InverseFast(self.body_matrix) * self.head_matrix

	# !! Call ONE TIME by frame !!
	def update(self):
		body_mtx = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(0, 0, 0))
		vr_state = hg.OpenVRGetState(body_mtx, 1, 1000)
		vs_left, vs_right = hg.OpenVRStateToViewState(vr_state)
		self.fov_left = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_left.proj))
		self.fov_right = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_right.proj))
		self.head_matrix = vr_state.head
		self.eye_left_offset = vr_state.left.offset
		self.eye_right_offset = vr_state.right.offset


class VRViewState:
	def __init__(self, camera:hg.Node, vr_view: VRState):
		self.z_near = 0
		self.z_far = 0
		self.head_matrix = None
		self.initial_head_matrix = None
		self.eye_left = None
		self.eye_right = None
		self.vs_left = None
		self.vs_right = None
		self.update(camera, vr_view)

	def update(self, camera, vr_view):
		cam = camera.GetCamera()
		self.z_near = cam.GetZNear()
		self.z_far = cam.GetZFar()

		# Compute current head matrix relative to initial_head_matrix
		local_head_matrix = hg.InverseFast(vr_view.initial_head_matrix) * vr_view.head_matrix

		# World head:
		self.head_matrix = camera.GetTransform().GetWorld() * local_head_matrix
		self.initial_head_matrix = camera.GetTransform().GetWorld()

		# World eyes:
		self.eye_left = self.head_matrix * vr_view.eye_left_offset
		self.eye_right = self.head_matrix * vr_view.eye_right_offset

		# View states:
		self.vs_left = hg.ComputePerspectiveViewState(self.eye_left, vr_view.fov_left, self.z_near, self.z_far, vr_view.ratio)
		self.vs_right = hg.ComputePerspectiveViewState(self.eye_right, vr_view.fov_right, self.z_near, self.z_far, vr_view.ratio)
class PlanetRender:
	def __init__(self, scene, resolution: hg.Vec2, terrain_position, terrain_offset):  # , pipeline_resource:hg.PipelineResource):

		# Vertex model:
		vs_decl = hg.VertexLayout()
		vs_decl.Begin()
		vs_decl.Add(hg.A_Position, 3, hg.AT_Float)
		vs_decl.Add(hg.A_Normal, 3, hg.AT_Uint8, True, True)
		vs_decl.End()

		self.uniforms_list = hg.UniformSetValueList()
		self.textures_list = hg.UniformSetTextureList()

		#self.quad_mdl = hg.CreateCubeModel(vs_decl, resolution.x / resolution.y * 2, 2, 0.01)
		self.quad_mdl = hg.CreatePlaneModel(vs_decl,  resolution.x / resolution.y * 2, 2, 1, 1)
		self.quad_matrix = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(-90), hg.Deg(0), hg.Deg(0)), hg.Vec3(1, 1, 1))
		self.sky_sea_render = hg.LoadProgramFromAssets("shaders/planet_render")

		self.scene = scene

		self.planet_radius = 6000000
		self.atmosphere_thickness = 100000
		self.atmosphere_falloff = 0.2
		self.atmosphere_luminosity_falloff = 0.2
		self.sun_light = scene.GetNode("Sun")

		#Atmosphere
		self.space_color = hg.Color(2 / 255, 2 / 255, 4 / 255, 1)
		self.high_atmosphere_color = hg.Color(17 / 255, 56 / 255, 155 / 255, 1)
		self.low_atmosphere_color = hg.Color(76 / 255, 128 / 255, 255 / 255, 1)
		self.horizon_line_color = hg.Color(1, 1, 1, 1)

		self.high_atmosphere_pos = 0.75
		self.low_atmosphere_pos = 0.5
		self.horizon_line_pos = 0.1
		self.horizon_line_falloff = 0.2

		self.horizon_low_line_size = 0.1
		self.horizon_low_line_falloff = 0.2

		self.tex_sky_intensity = 1
		self.tex_space_intensity = 0

		# clouds
		self.clouds_scale = hg.Vec3(50000., 0.117, 40000.)
		self.clouds_altitude = 3000
		self.clouds_absorption = 0.145

		# Sun

		self.sun_size = 2
		self.sun_smooth = 0.1
		self.sun_glow = 1
		self.sun_space_glow_intensity = 0.25

		# Sea
		self.sea_color = hg.Color(19 / 255, 39 / 255, 89 / 255, 1)
		self.underwater_color = hg.Color(76 / 255, 128 / 255, 255 / 255, 1)
		self.sea_reflection = 0.5
		self.reflect_color = hg.Color(0.464, 0.620, 1, 1)
		self.scene_reflect = 0
		self.sea_scale = hg.Vec3(0.02, 16, 0.005)
		self.render_sea = 1
		self.render_scene_reflection = False
		self.reflect_map = None
		self.reflect_map_depth = None
		self.reflect_offset = 50

		# Terrain
		self.terrain_scale = hg.Vec3(41480, 1000, 19587)
		self.terrain_position = terrain_position + terrain_offset
		self.terrain_intensity = 0.5
		self.terrain_clamp = 0.01
		self.terrain_coast_edges = hg.Vec2(0.1, 0.3)

		# Textures:
		self.sea_texture = hg.LoadTextureFromAssets("textures/ocean_noises.png", 0)[0]
		self.stream_texture = hg.LoadTextureFromAssets("textures/stream.png", 0)[0]
		self.clouds_map = hg.LoadTextureFromAssets("textures/clouds_map.png", 0)[0]
		self.tex_sky = hg.LoadTextureFromAssets("textures/clouds.png", 0)[0]
		self.tex_space = hg.LoadTextureFromAssets("textures/8k_stars_milky_way.jpg", 0)[0]
		self.tex_terrain = hg.LoadTextureFromAssets("island_chain/textureRVBA.png", 0)[0]

	def gui(self, cam_pos):
		if hg.ImGuiBegin("Sea & Sky render Settings"):

			hg.ImGuiSetWindowPos("Sea & Sky render Settings", hg.Vec2(10, 390), hg.ImGuiCond_Once)
			hg.ImGuiSetWindowSize("Sea & Sky render Settings", hg.Vec2(650, 600), hg.ImGuiCond_Once)

			d, f = hg.ImGuiDragVec3("Camera pos", cam_pos, 100)
			if d: cam_pos = f

			if hg.ImGuiButton("Load sea parameters"):
				self.load_json_script()
			hg.ImGuiSameLine()
			if hg.ImGuiButton("Save sea parameters"):
				self.save_json_script()

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("Planet radius", self.planet_radius, 1000, 6000000)
			if d: self.planet_radius = f
			d, f = hg.ImGuiInputFloat("Atmosphere thickness", self.atmosphere_thickness)
			if d: self.atmosphere_thickness = f

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("Sky texture intensity", self.tex_sky_intensity, 0, 1)
			if d: self.tex_sky_intensity = f
			d, f = hg.ImGuiSliderFloat("Space texture intensity", self.tex_space_intensity, 0, 1)
			if d: self.tex_space_intensity = f

			hg.ImGuiSeparator()

			f, c = hg.ImGuiColorEdit("Space color", self.space_color)
			if f: self.space_color = c
			f, c = hg.ImGuiColorEdit("High atmosphere color", self.high_atmosphere_color)
			if f: self.high_atmosphere_color = c
			f, c = hg.ImGuiColorEdit("Low atmosphere color", self.low_atmosphere_color)
			if f: self.low_atmosphere_color = c
			f, c = hg.ImGuiColorEdit("Horizon line color", self.horizon_line_color)
			if f: self.horizon_line_color = c

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("Atmosphere falloff", self.atmosphere_falloff, 0, 2)
			if d: self.atmosphere_falloff = f
			d, f = hg.ImGuiSliderFloat("Atmosphere luminosity falloff", self.atmosphere_luminosity_falloff, 0, 2)
			if d: self.atmosphere_luminosity_falloff = f
			d, f = hg.ImGuiSliderFloat("High atmosphere pos", self.high_atmosphere_pos, 0, 1)
			if d: self.high_atmosphere_pos = f
			d, f = hg.ImGuiSliderFloat("Low atmosphere pos", self.low_atmosphere_pos, 0, 1)
			if d: self.low_atmosphere_pos = f
			d, f = hg.ImGuiSliderFloat("horizon line falloff", self.horizon_line_pos, 0, 1)
			if d: self.horizon_line_pos = f
			d, f = hg.ImGuiSliderFloat("horizon line pos", self.horizon_line_falloff, 0, 2)
			if d: self.horizon_line_falloff = f

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("horizon low line size", self.horizon_low_line_size, 0, 2)
			if d: self.horizon_low_line_size = f
			d, f = hg.ImGuiSliderFloat("horizon low line falloff", self.horizon_low_line_falloff, 0, 2)
			if d: self.horizon_low_line_falloff = f

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("Sun size", self.sun_size, 0.1, 10)
			if d: self.sun_size = f
			d, f = hg.ImGuiSliderFloat("Sun smooth", self.sun_smooth, 0, 10)
			if d: self.sun_smooth = f
			d, f = hg.ImGuiSliderFloat("Sun glow intensity", self.sun_glow, 0, 1)
			if d: self.sun_glow = f
			d, f = hg.ImGuiSliderFloat("Sun space glow intensity", self.sun_space_glow_intensity, 0, 1)
			if d: self.sun_space_glow_intensity = f

			hg.ImGuiSeparator()

			f, c = hg.ImGuiColorEdit("Near water color", self.sea_color)
			if f: self.sea_color = c
			f, c = hg.ImGuiColorEdit("Underwater color", self.underwater_color)
			if f: self.underwater_color = c
			f, c = hg.ImGuiColorEdit("Reflect color", self.reflect_color)
			if f: self.reflect_color = c

			hg.ImGuiSeparator()

			d, f = hg.ImGuiDragVec3("Clouds scale", self.clouds_scale, 1)
			if d: self.clouds_scale = f
			d, f = hg.ImGuiDragFloat("Clouds altitude", self.clouds_altitude, 0.1, 1, 100000)
			if d: self.clouds_altitude = f
			d, f = hg.ImGuiSliderFloat("Clouds absoption", self.clouds_absorption, 0, 1)
			if d: self.clouds_absorption = f

			hg.ImGuiSeparator()

			d, f = hg.ImGuiCheckbox("Water reflection", self.render_scene_reflection)
			if d:
				self.render_scene_reflection = f
				if self.render_scene_reflection:
					self.scene_reflect = 1
				else:
					self.scene_reflect = 0

			d, f = hg.ImGuiDragVec3("sea scale", self.sea_scale, 1)
			if d: self.sea_scale = f
			# Not implemented yet !
			d, f = hg.ImGuiSliderFloat("Sea reflection", self.sea_reflection, 0, 1)
			if d: self.sea_reflection = f
			d, f = hg.ImGuiSliderFloat("Reflect offset", self.reflect_offset, 1, 1000)
			if d: self.reflect_offset = f

			hg.ImGuiSeparator()

			d, f = hg.ImGuiSliderFloat("Terrain texture intensity", self.terrain_intensity, 0, 1)
			if d: self.terrain_intensity = f
			d, f = hg.ImGuiDragVec3("Terrain scale", self.terrain_scale, 1)
			if d: self.terrain_scale = f
			d, f = hg.ImGuiDragVec3("Terrain position", self.terrain_position, 1)
			if d: self.terrain_position = f
			d, f = hg.ImGuiDragVec2("Terrain coast_edges", self.terrain_coast_edges, 0.001)
			if d: self.terrain_coast_edges = f
			d, f = hg.ImGuiSliderFloat("Terrain clamp", self.terrain_clamp, 0, 1)
			if d: self.terrain_clamp = f


		hg.ImGuiEnd()
		return cam_pos

	def load_json_script(self, file_name="scripts/planet_parameters.json"):
		file = hg.OpenText(file_name)
		if not file:
			print("ERROR - Can't open json file : " + file_name)
		else:
			json_script = hg.ReadString(file)
			hg.Close(file)
			if json_script != "":
				script_parameters = json.loads(json_script)
				self.low_atmosphere_color = list_to_color(script_parameters["low_atmosphere_color"])
				self.high_atmosphere_color = list_to_color(script_parameters["high_atmosphere_color"])
				self.space_color = list_to_color(script_parameters["space_color"])
				self.horizon_line_color = list_to_color(script_parameters["horizon_line_color"])

				self.sun_size = script_parameters["sun_size"]
				self.sun_smooth = script_parameters["sun_smooth"]
				self.sun_glow = script_parameters["sun_glow"]
				self.sun_space_glow_intensity = script_parameters["sun_space_glow_intensity"]

				self.planet_radius = script_parameters["planet_radius"]
				self.atmosphere_thickness = script_parameters["atmosphere_thickness"]
				self.atmosphere_falloff = script_parameters["atmosphere_falloff"]
				self.atmosphere_luminosity_falloff = script_parameters["atmosphere_luminosity_falloff"]
				self.high_atmosphere_pos = script_parameters["high_atmosphere_pos"]
				self.low_atmosphere_pos = script_parameters["low_atmosphere_pos"]
				self.horizon_line_pos = script_parameters["horizon_line_pos"]
				self.horizon_line_falloff = script_parameters["horizon_line_falloff"]

				self.horizon_low_line_size = script_parameters["horizon_low_line_size"]
				self.horizon_low_line_falloff = script_parameters["horizon_low_line_falloff"]

				self.tex_sky_intensity = script_parameters["tex_sky_intensity"]
				self.tex_space_intensity = script_parameters["tex_space_intensity"]

				self.underwater_color = list_to_color(script_parameters["underwater_color"])
				self.sea_color = list_to_color(script_parameters["sea_color"])
				self.sea_scale = list_to_vec3(script_parameters["sea_scale"])
				self.sea_reflection = script_parameters["sea_reflection"]
				self.reflect_offset = script_parameters["reflect_offset"]
				self.scene_reflect = script_parameters["scene_reflect"]
				self.reflect_color = list_to_color(script_parameters["reflect_color"])

				self.terrain_intensity = script_parameters["terrain_intensity"]
				self.terrain_scale = list_to_vec3(script_parameters["terrain_scale"])
				self.terrain_coast_edges = list_to_vec2(script_parameters["terrain_coast_edges"])
				self.terrain_clamp = script_parameters["terrain_clamp"]
				self.terrain_position = list_to_vec3(script_parameters["terrain_position"])

				self.clouds_scale = list_to_vec3(script_parameters["clouds_scale"])
				self.clouds_altitude = script_parameters["clouds_altitude"]
				self.clouds_absorption = script_parameters["clouds_absorption"]

				if self.scene_reflect > 0.5:
					self.render_scene_reflection = True
				else:
					self.render_scene_reflection = False

	def save_json_script(self, output_filename="scripts/planet_parameters.json"):
		script_parameters = {
							"space_color": color_to_list(self.space_color),
							"high_atmosphere_color": color_to_list(self.high_atmosphere_color),
							"high_atmosphere_pos": self.high_atmosphere_pos,
							"low_atmosphere_color": color_to_list(self.low_atmosphere_color),
							"low_atmosphere_pos": self.low_atmosphere_pos,
							"horizon_line_color": color_to_list(self.horizon_line_color),
							"horizon_line_pos": self.horizon_line_pos,
							"horizon_line_falloff": self.horizon_line_falloff,

							"horizon_low_line_falloff": self.horizon_low_line_falloff,
							"horizon_low_line_size": self.horizon_low_line_size,

							"tex_sky_intensity": self.tex_sky_intensity,
							"tex_space_intensity": self.tex_space_intensity,

							"clouds_scale": vec3_to_list(self.clouds_scale),
							"clouds_altitude": self.clouds_altitude,
							"clouds_absorption": self.clouds_absorption,

							"sea_color": color_to_list(self.sea_color),
							"underwater_color": color_to_list(self.underwater_color),
							"reflect_color": color_to_list(self.reflect_color),
							"sea_reflection": self.sea_reflection,
							"sea_scale": vec3_to_list(self.sea_scale),
							"reflect_offset": self.reflect_offset,
							"scene_reflect": self.scene_reflect,

							"sun_size": self.sun_size,
							"sun_smooth": self.sun_smooth,
							"sun_glow": self.sun_glow,
							"sun_space_glow_intensity": self.sun_space_glow_intensity,

							"terrain_intensity": self.terrain_intensity,
							"terrain_scale": vec3_to_list(self.terrain_scale),
							"terrain_clamp": self.terrain_clamp,
							"terrain_coast_edges": vec2_to_list(self.terrain_coast_edges),
							"terrain_position": vec3_to_list(self.terrain_position),

							"planet_radius": self.planet_radius,
							"atmosphere_thickness": self.atmosphere_thickness,
							"atmosphere_falloff": self.atmosphere_falloff,
							"atmosphere_luminosity_falloff": self.atmosphere_luminosity_falloff
		}
		json_script = json.dumps(script_parameters, indent=4)
		file = hg.OpenWrite(output_filename)
		if file:
			hg.WriteString(file, json_script)
			hg.Close(file)
			return True
		else:
			print("ERROR - Can't open json file : " + output_filename)
			return False

	def get_distance(self, r, altitude, angle):
		return (r + altitude) * cos(angle) - r * sin(acos(sin(angle) * (r + altitude) / r))

	def get_distance_far(self, r, altitude, angle):
		return -self.get_distance(r, altitude, angle - pi)

	def get_horizon_angle(self,r , altitude):
		alt = max(0, altitude)
		return (pi / 2.0) - atan(sqrt(alt * (alt + 2.0 * r)) / r)

	def get_atmosphere_view_thickness_max(self, horizon_angle, altitude):
		dmin = self.get_distance(self.planet_radius + self.atmosphere_thickness, altitude - self.atmosphere_thickness, horizon_angle)
		dmax = self.get_distance_far(self.planet_radius + self.atmosphere_thickness, altitude - self.atmosphere_thickness, horizon_angle)
		#print("dmin %f, dmax %f" % (dmin,dmax))
		if dmin <= 0:
			return dmax
		else:
			return dmax - dmin

	def get_atmosphere_angle(self, altitude, horizon_angle):
		f = altitude / max(1,self.atmosphere_thickness)
		if f<=1:
			return pi * (1 - f) + (pi / 2 * f) - horizon_angle
		else:
			return self.get_horizon_angle(self.planet_radius + self.atmosphere_thickness, altitude - self.atmosphere_thickness) - horizon_angle

	def get_atmosphere_size(self,altitude, size):
		return size * exp(-pow(altitude/self.atmosphere_thickness * 0.888, 2))

	def render_vr(self, view_id, vr_state: hg.OpenVRState, vs_left: hg.ViewState, vs_right: hg.ViewState, vr_left_fb, vr_right_fb, reflect_texture_left: hg.Texture = None, reflect_depth_texture_left: hg.Texture = None, reflect_texture_right: hg.Texture = None, reflect_depth_texture_right: hg.Texture = None):

		vr_resolution = hg.Vec2(vr_state.width, vr_state.height)
		eye_left = vr_state.head * vr_state.left.offset
		eye_right = vr_state.head * vr_state.right.offset

		focal_distance_left = hg.ExtractZoomFactorFromProjectionMatrix(vs_left.proj)
		focal_distance_right = hg.ExtractZoomFactorFromProjectionMatrix(vs_right.proj)

		cam_normal_left = hg.GetRotationMatrix(eye_left)
		cam_normal_right = hg.GetRotationMatrix(eye_right)
		cam_pos_left = hg.GetT(eye_left)
		cam_pos_right = hg.GetT(eye_right)

		# Left:
		z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_left.proj)
		z_ratio = (z_near + 0.01) / focal_distance_left
		self.update_shader(cam_pos_left, cam_normal_left, focal_distance_left, z_near, z_far, vr_resolution, reflect_texture_left, reflect_depth_texture_left)
		hg.SetViewFrameBuffer(view_id, vr_left_fb.GetHandle())
		hg.SetViewRect(view_id, 0, 0, int(vr_resolution.x), int(vr_resolution.y))
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		hg.SetViewTransform(view_id, hg.InverseFast(vr_state.left.offset), vs_left.proj)
		matrx = vr_state.left.offset * hg.TransformationMat4(hg.Vec3(0, 0, focal_distance_left * z_ratio), hg.Vec3(hg.Deg(-90), hg.Deg(0), hg.Deg(0)), hg.Vec3(1, 1, 1) * z_ratio)
		hg.DrawModel(view_id, self.quad_mdl, self.sky_sea_render, self.uniforms_list, self.textures_list, matrx)
		view_id += 1

		#Right
		z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_right.proj)
		z_ratio = (z_near + 0.01) / focal_distance_right
		self.update_shader(cam_pos_right, cam_normal_right, focal_distance_right, z_near, z_far, vr_resolution, reflect_texture_right, reflect_depth_texture_right)
		hg.SetViewFrameBuffer(view_id, vr_right_fb.GetHandle())
		hg.SetViewRect(view_id, 0, 0, int(vr_resolution.x), int(vr_resolution.y))
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		hg.SetViewTransform(view_id, hg.InverseFast(vr_state.right.offset), vs_right.proj)
		matrx = vr_state.right.offset * hg.TransformationMat4(hg.Vec3(0, 0, focal_distance_right * z_ratio), hg.Vec3(hg.Deg(-90), hg.Deg(0), hg.Deg(0)), hg.Vec3(1, 1, 1) * z_ratio)
		hg.DrawModel(view_id, self.quad_mdl, self.sky_sea_render, self.uniforms_list, self.textures_list, matrx)

		return view_id + 1

	def update_shader(self, cam_pos, cam_normal, focal_distance, z_near, z_far, resolution, reflect_texture: hg.Texture = None, reflect_depth_texture: hg.Texture = None):

		horizon_angle = self.get_horizon_angle(self.planet_radius, cam_pos.y)
		atmosphere_angle = self.get_atmosphere_angle(cam_pos.y, horizon_angle)

		self.uniforms_list.clear()

		self.uniforms_list.push_back(hg.MakeUniformSetValue("resolution", hg.Vec4(resolution.x, resolution.y, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("focal_distance", hg.Vec4(focal_distance, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("cam_position", hg.Vec4(cam_pos.x, cam_pos.y, cam_pos.z, cam_pos.y / max(1,self.atmosphere_thickness))))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("cam_normal", cam_normal))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("z_Frustum", hg.Vec4(z_near, z_far, 0, 0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("horizon_angle", hg.Vec4(horizon_angle, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("planet_radius", hg.Vec4(self.planet_radius, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("atmosphere_params", hg.Vec4(self.atmosphere_thickness, self.atmosphere_falloff, self.atmosphere_luminosity_falloff, atmosphere_angle)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("high_atmosphere_pos", hg.Vec4(self.high_atmosphere_pos, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("low_atmosphere_pos", hg.Vec4(self.low_atmosphere_pos, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("horizon_line_params", hg.Vec4(self.horizon_line_pos, self.horizon_line_falloff, 0, 0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("horizon_low_line_params", hg.Vec4(self.horizon_low_line_size, self.horizon_low_line_falloff,0,0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("clouds_scale", hg.Vec4(1 / self.clouds_scale.x, self.clouds_scale.y, 1 / self.clouds_scale.z, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("clouds_altitude", hg.Vec4(self.clouds_altitude, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("clouds_absorption", hg.Vec4(self.clouds_absorption, 0, 0, 0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("tex_sky_intensity", hg.Vec4(self.tex_sky_intensity, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("tex_space_intensity", hg.Vec4(self.tex_space_intensity, 0, 0, 0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("sea_scale", hg.Vec4(self.sea_scale.x, self.sea_scale.y, self.sea_scale.z, 0)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("terrain_scale", hg.Vec4(self.terrain_scale.x, self.terrain_scale.y, -self.terrain_scale.z, self.terrain_intensity)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("terrain_position", hg.Vec4(self.terrain_position.x, self.terrain_position.y, self.terrain_position.z, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("terrain_edges", hg.Vec4(self.terrain_coast_edges.x, self.terrain_coast_edges.y, self.terrain_clamp, 0)))

		# Colors:
		colors_uniforms = {"space_color", "high_atmosphere_color", "low_atmosphere_color", "horizon_line_color", "sea_color", "underwater_color", "reflect_color"}
		for cn in colors_uniforms:
			self.uniforms_list.push_back(hg.MakeUniformSetValue(cn, hg.Vec4(getattr(self, cn).r, getattr(self, cn).g, getattr(self, cn).b, getattr(self, cn).a)))

		amb = self.scene.environment.ambient
		self.uniforms_list.push_back(hg.MakeUniformSetValue("ambient_color", hg.Vec4(amb.r, amb.g, amb.b, amb.a)))

		sun_color = self.sun_light.GetLight().GetDiffuseColor()
		sun_dir = hg.GetZ(self.sun_light.GetTransform().GetWorld())
		self.uniforms_list.push_back(hg.MakeUniformSetValue("sun_color", hg.Vec4(sun_color.r, sun_color.g, sun_color.b, sun_color.a)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("sun_dir", hg.Vec4(sun_dir.x, sun_dir.y, sun_dir.z, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("sun_params", hg.Vec4(self.sun_size, self.sun_smooth, self.sun_glow, self.sun_space_glow_intensity)))

		self.uniforms_list.push_back(hg.MakeUniformSetValue("reflect_offset", hg.Vec4(self.reflect_offset, 0, 0, 0)))
		if reflect_texture is None or reflect_depth_texture is None:
			flag_reflect = 0
		else:
			flag_reflect = self.scene_reflect
		self.uniforms_list.push_back(hg.MakeUniformSetValue("scene_reflect", hg.Vec4(flag_reflect, 0, 0, 0)))
		self.uniforms_list.push_back(hg.MakeUniformSetValue("sea_reflection", hg.Vec4(self.sea_reflection, 0, 0, 0)))

		# --- Setup textures:

		self.textures_list.clear()
		self.textures_list.push_back(hg.MakeUniformSetTexture("sea_noises", self.sea_texture, 0))
		self.textures_list.push_back(hg.MakeUniformSetTexture("stream_texture", self.stream_texture, 1))
		self.textures_list.push_back(hg.MakeUniformSetTexture("clouds_map", self.clouds_map, 2))
		self.textures_list.push_back(hg.MakeUniformSetTexture("tex_sky", self.tex_sky, 3))
		self.textures_list.push_back(hg.MakeUniformSetTexture("tex_space", self.tex_space, 4))
		if reflect_texture is not None:
			self.textures_list.push_back(hg.MakeUniformSetTexture("reflect_map", reflect_texture, 5))
		if reflect_depth_texture is not None:
			self.textures_list.push_back(hg.MakeUniformSetTexture("reflect_map_depth", reflect_depth_texture, 6))
		self.textures_list.push_back(hg.MakeUniformSetTexture("terrain_map", self.tex_terrain, 7))


	def render(self, view_id, camera: hg.Node, resolution: hg.Vec2, reflect_texture: hg.Texture = None, reflect_depth_texture: hg.Texture = None, frame_buffer=None):

		# Vars:
		cam = camera.GetCamera()
		if cam.GetIsOrthographic():
			focal_distance = camera.GetTransform().GetPos().y / (cam.GetSize() / 2)
		else:
			focal_distance = hg.FovToZoomFactor(cam.GetFov())
		cam_mat = camera.GetTransform().GetWorld()

		cam_normal = hg.GetRotationMatrix(cam_mat)
		cam_pos = hg.GetT(cam_mat)
		z_near = cam.GetZNear()
		z_far = cam.GetZFar()

		self.update_shader(cam_pos, cam_normal, focal_distance, z_near, z_far, resolution, reflect_texture, reflect_depth_texture)

		# --- Set View:
		if frame_buffer is not None:
			hg.SetViewFrameBuffer(view_id, frame_buffer.handle)
		hg.SetViewRect(view_id, 0, 0, int(resolution.x), int(resolution.y))
		#hg.SetViewClear(view_id, 0, 0x0, 1.0, 0)
		hg.SetViewClear(view_id, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
		cam = hg.TransformationMat4(hg.Vec3(0, 0, -focal_distance), hg.Vec3(0, 0, 0))

		view = hg.InverseFast(cam)
		proj = hg.ComputePerspectiveProjectionMatrix(0.1, 100, focal_distance, hg.Vec2(resolution.x / resolution.y, 1))
		hg.SetViewTransform(view_id, view, proj)
		hg.DrawModel(view_id, self.quad_mdl, self.sky_sea_render, self.uniforms_list, self.textures_list, self.quad_matrix)
		return view_id + 1
class WaterReflection():
	def __init__(self, scene, resolution: hg.Vec2, antialiasing = 4, flag_vr=False):
		self.flag_vr = flag_vr
		# Parameters:
		self.color = hg.Color(1, 0, 0, 1)
		self.reflect_level = 0.75

		self.camera_reflect = hg.CreateCamera(scene, hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(0, 0, 0)), 1, 10000)
		self.main_camera = None

		self.render_program = hg.LoadProgramFromAssets("shaders/copy")

		if not flag_vr:
			self.quad_frameBuffer = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_reflect")
		else:
			self.quad_frameBuffer_left = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_reflect_left")
			self.quad_frameBuffer_right = hg.CreateFrameBuffer(int(resolution.x), int(resolution.y), hg.TF_RGBA8, hg.TF_D32F, antialiasing, "frameBuffer_reflect_right")

	@staticmethod
	def get_plane_projection_factor(p: hg.Vec3, plane_origine: hg.Vec3, plane_normal: hg.Vec3):
		d = -plane_normal.x * plane_origine.x - plane_normal.y * plane_origine.y - plane_normal.z * plane_origine.z
		return -plane_normal.x * p.x - plane_normal.y * p.y - plane_normal.z * p.z - d

	def compute_reflect_matrix(self, mat):
		plane_pos = hg.Vec3(0, 0, 0)
		plane_normal = hg.Vec3(0, 1, 0)
		pos = hg.GetT(mat)
		t = self.get_plane_projection_factor(pos, plane_pos, plane_normal)
		pos_reflect = pos + plane_normal * 2 * t
		xAxis = hg.GetX(mat)
		zAxis = hg.GetZ(mat)
		px = pos + xAxis
		tx = self.get_plane_projection_factor(px, plane_pos, plane_normal)
		x_reflect = px + plane_normal * 2 * tx - pos_reflect
		z_reflect = hg.Reflect(zAxis, plane_normal)
		y_reflect = hg.Cross(z_reflect, x_reflect)
		mat_reflect = hg.TranslationMat4(pos_reflect)
		hg.SetX(mat_reflect, x_reflect)
		hg.SetY(mat_reflect, y_reflect)
		hg.SetZ(mat_reflect, z_reflect)
		return mat_reflect

	def set_camera(self, scene):
		self.main_camera = scene.GetCurrentCamera()
		mat_reflect = self.compute_reflect_matrix(self.main_camera.GetTransform().GetWorld())
		self.camera_reflect.GetTransform().SetWorld(mat_reflect)
		cam_org = self.main_camera.GetCamera()
		cam = self.camera_reflect.GetCamera()
		cam.SetFov(cam_org.GetFov())
		cam.SetZNear(cam_org.GetZNear())
		cam.SetZFar(cam_org.GetZFar())
		scene.SetCurrentCamera(self.camera_reflect)

	def compute_vr_reflect(self, camera, vr_state: hg.OpenVRState, vs_left: hg.ViewState, vs_right: hg.ViewState):

		eye_left = vr_state.head * vr_state.left.offset
		eye_right = vr_state.head * vr_state.right.offset

		mat_left_reflect = self.compute_reflect_matrix(eye_left)
		mat_right_reflect = self.compute_reflect_matrix(eye_right)

		fov_left = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_left.proj))
		fov_right = hg.ZoomFactorToFov(hg.ExtractZoomFactorFromProjectionMatrix(vs_right.proj))

		znear = camera.GetCamera().GetZNear()
		zfar = camera.GetCamera().GetZFar()

		ratio = hg.Vec2(vr_state.width / vr_state.height, 1)

		vs_left_reflect = hg.ComputePerspectiveViewState(mat_left_reflect, fov_left, znear, zfar, ratio)
		vs_right_reflect = hg.ComputePerspectiveViewState(mat_right_reflect, fov_right, znear, zfar, ratio)

		return vs_left_reflect, vs_right_reflect

	def restore_camera(self, scene):
		scene.SetCurrentCamera(self.main_camera)

	def load_parameters(self, file_name="assets/scripts/water_reflection.json"):
		json_script = hg.GetFilesystem().FileToString(file_name)
		if json_script != "":
			script_parameters = json.loads(json_script)
			self.color = list_to_color(script_parameters["color"])
			self.reflect_level = script_parameters["reflect_level"]

	def save_parameters(self, output_filename="assets/scripts/water_reflection.json"):
		script_parameters = {"color": color_to_list(self.color), "reflect_level": self.reflect_level}
		json_script = json.dumps(script_parameters, indent=4)
		return hg.GetFilesystem().StringToFile(output_filename, json_script)
class Main:

    # Main display configuration (user-defined in "config.json" file)

    flag_fullscreen = False
    resolution = hg.Vec2(1920, 1080)
    flag_shadowmap = True
    flag_OpenGL = False
    antialiasing = 4
    flag_display_HUD = True

    # Control devices

    control_mode = ControlDevice.CM_KEYBOARD

    # VR mode
    flag_vr = False
    vr_left_fb = None
    vr_right_fb = None
    # VR screen display
    vr_quad_layout = None
    vr_quad_model = None
    vr_quad_render_state = None
    eye_t_x = 0
    vr_quad_matrix = None
    vr_tex0_program = None
    vr_quad_uniform_set_value_list = None
    vr_quad_uniform_set_texture_list = None
    vr_hud = None # vec3 width, height, zdistance

    vr_state = None # OpenVRState

    initial_head_matrix = None

    #
    flag_exit = False
    win = None

    timestamp = 0  # Frame count.
    timestep = 1 / 60  # Frame dt

    flag_network_mode = False
    flag_client_update_mode = False
    flag_client_connected = False
    flag_client_ask_update_scene = False

    flag_renderless = False
    flag_running = False
    flag_display_radar_in_renderless = True
    frame_time = 0 # Used to synchronize Renderless display informations
    flag_activate_particles_mem = True
    flag_sfx_mem = True
    max_view_id = 0

    flag_paddle = False
    flag_generic_controller = False

    assets_compiled = "assets_compiled"

    allies_missiles_smoke_color = hg.Color(1.0, 1.0, 1.0, 1.0)
    ennemies_missiles_smoke_color = hg.Color(1.0, 1.0, 1.0, 1.0)

    flag_sfx = True
    flag_control_views = True
    flag_display_fps = False
    flag_display_landing_trajectories = False
    flag_display_selected_aircraft = False
    flag_display_machines_bounding_boxes = False
    flag_display_physics_debug = False
    nfps = [0] * 100
    nfps_i = 0
    num_fps = 0

    post_process = None
    render_data = None
    scene = None
    scene_physics = None
    clocks = None

    flag_gui = False

    sea_render = None
    water_reflexion = None

    num_start_frames = 10
    keyboard = None
    mouse = None
    gamepad = None
    generic_controller = None
    pipeline = None

    current_state = None
    t = 0
    fading_to_next_state = False
    end_state_timer = 0
    end_phase_following_aircraft = None

    current_view = None
    camera = None
    camera_fps = None

    intro_anim_id = 2
    camera_intro = None
    anim_camera_intro_dist = None
    anim_camera_intro_rot = None
    display_dark_design = True
    display_logo = True

    satellite_camera = None
    satellite_view = False
    aircraft = None
    ennemy_aircraft = None
    user_aircraft = None
    player_view_mode = SmartCamera.TYPE_TRACKING

    aircraft_carrier_allies = []
    aircraft_carrier_ennemies = []

    missile_launchers_allies = []
    missile_launchers_ennemies = []

    smart_camera = None

    pl_resources = None

    background_color = 0x1070a0ff  # 0xb9efffff

    ennemyaircraft_nodes = None
    num_players_ennemies = 0
    num_players_allies = 0
    num_missile_launchers_allies = 0
    num_missile_launchers_ennemies = 0
    players_allies = []
    players_ennemies = []
    players_sfx = []
    missiles_allies = []
    missiles_ennemies = []
    missiles_sfx = []
    views_carousel = []
    views_carousel_ptr = 0

    sprites_display_list = []
    #texts_display_list = []

    destroyables_list = []  # whole missiles, aircrafts, ships used by HUD radar
    destroyables_items = {} # items stored by their names

    font_program = None
    title_font_path = "font/destroy.ttf"
    hud_font_path = "font/Furore.otf"
    title_font = None
    hud_font = None
    text_matrx = None
    text_uniform_set_values = hg.UniformSetValueList()
    text_uniform_set_texture_list = hg.UniformSetTextureList()
    text_render_state = None

    fading_cptr = 0
    menu_fading_cptr = 0

    spr_design_menu = None
    spr_logo = None

    main_music_ref = []
    main_music_state = []
    main_music_source = []

    master_sfx_volume = 0

    # Cockpit view:
    flag_cockpit_view = False
    
    
    scene_cockpit = None
    scene_cockpit_frameBuffer = None
    scene_cockpit_frameBuffer_left = None
    scene_cockpit_frameBuffer_right = None
    scene_cockpit_aircrafts = []  # Aircrafts models used for cockpit view (1 model by aircraft type)
    user_cockpit_aircraft = None
    cockpit_scene_quad_model = None
    cockpit_scene_quad_uniform_set_value_list = None
    cockpit_scene_quad_uniform_set_texture_list = None

    #======= Aircrafts view:
    selected_aircraft_id = 0
    selected_aircraft = None

    @classmethod
    def init(cls):
        cls.pl_resources = hg.PipelineResources()
        cls.keyboard = hg.Keyboard()
        cls.mouse = hg.Mouse()
        cls.gamepad = hg.Gamepad()
        Overlays.init()
        ControlDevice.init(cls.keyboard, cls.mouse, cls.gamepad, cls.generic_controller)

    @classmethod
    def setup_vr(cls):
        if not hg.OpenVRInit():
            return False

        cls.vr_left_fb = hg.OpenVRCreateEyeFrameBuffer(hg.OVRAA_MSAA4x)
        cls.vr_right_fb = hg.OpenVRCreateEyeFrameBuffer(hg.OVRAA_MSAA4x)

        body_mtx = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(0, 0, 0))
        cls.vr_state = hg.OpenVRGetState(body_mtx, 1, 1000)
        cls.vr_resolution = hg.Vec2(cls.vr_state.width, cls.vr_state.height)

        cls.update_initial_head_matrix(cls.vr_state)

        # Setup vr screen display:

        cls.vr_quad_layout = hg.VertexLayout()
        cls.vr_quad_layout.Begin().Add(hg.A_Position, 3, hg.AT_Float).Add(hg.A_TexCoord0, 3, hg.AT_Float).End()

        cls.vr_quad_model = hg.CreatePlaneModel(cls.vr_quad_layout, 1, 1, 1, 1)
        cls.vr_quad_render_state = hg.ComputeRenderState(hg.BM_Alpha, hg.DT_Disabled, hg.FC_Disabled)

        eye_t_size = cls.resolution.x / 2.5
        cls.eye_t_x = (cls.resolution.x - 2 * eye_t_size) / 6 + eye_t_size / 2
        cls.vr_quad_matrix = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(90), hg.Deg(0), hg.Deg(0)), hg.Vec3(eye_t_size, 1, eye_t_size))

        cls.vr_tex0_program = hg.LoadProgramFromAssets("shaders/vrdisplay")

        cls.vr_quad_uniform_set_value_list = hg.UniformSetValueList()
        cls.vr_quad_uniform_set_value_list.clear()
        cls.vr_quad_uniform_set_value_list.push_back(hg.MakeUniformSetValue("color", hg.Vec4(1, 1, 1, 1)))

        cls.vr_quad_uniform_set_texture_list = hg.UniformSetTextureList()

        ratio = cls.resolution.x / cls.resolution.y
        size = 10
        cls.vr_hud = hg.Vec3(size * ratio, size, 12)

        return True

    @classmethod
    def init_game(cls):
        cls.init()
        cls.render_data = hg.SceneForwardPipelineRenderData()

        cls.scene = hg.Scene()
        cls.scene_physics = hg.SceneBullet3Physics()
        cls.clocks = hg.SceneClocks()
        cls.scene_physics.StepSimulation(hg.time_from_sec_f(1 / 60))

        hg.LoadSceneFromAssets("main.scn", cls.scene, cls.pl_resources, hg.GetForwardPipelineInfo())
        Destroyable_Machine.world_node = cls.scene.GetNode("world_node")

        # Remove Dummies objects:

        nl = cls.scene.GetAllNodes()
        num = nl.size()
        for i in range(num):
            nd = nl.at(i)
            node_name = nd.GetName()
            if node_name.split("_")[0] == "dummy":
                nd.RemoveObject()
                cls.scene.GarbageCollect()

        # print("GARBAGE: "+str(cls.scene.GarbageCollect()))

        cls.camera_cokpit = cls.scene.GetNode("Camera_cokpit")
        cls.camera = cls.scene.GetNode("Camera_follow")
        cls.camera_fps = cls.scene.GetNode("Camera_fps")
        cls.satellite_camera = cls.scene.GetNode("Camera_satellite")
        cls.smart_camera = SmartCamera(SmartCamera.TYPE_FOLLOW, cls.keyboard, cls.mouse)
        #  Camera used in start phase :
        cls.camera_intro = cls.scene.GetNode("Camera_intro")

        # Shadows setup
        sun = cls.scene.GetNode("Sun")
        if cls.flag_shadowmap:
            sun.GetLight().SetShadowType(hg.LST_Map)
        else:
            sun.GetLight().SetShadowType(hg.LST_None)

        if cls.flag_vr:
            framebuffers_resolution = cls.vr_resolution
        else:
            framebuffers_resolution = cls.resolution

        # ---------- Post process setup

        cls.post_process = PostProcess(framebuffers_resolution, cls.antialiasing, cls.flag_vr)

        # ---------- Destroyable machines original nodes lists:

        F14.init(cls.scene)
        F14_2.init(cls.scene)
        Rafale.init(cls.scene)
        Sidewinder.init(cls.scene)
        Meteor.init(cls.scene)
        Mica.init(cls.scene)

        # -------------- Sprites:
        Sprite.init_system()

        HUD.init(cls.resolution)
        HUD_Radar.init(cls.resolution)
        HUD_MachineGun.init(cls.resolution)
        HUD_MissileTarget.init(cls.resolution)

        cls.spr_design_menu = Sprite(1280, 720, "sprites/design_menu_b.png")
        cls.spr_design_menu.set_size(cls.resolution.x / 1280)

        cls.spr_logo = Sprite(1920, 1080, "sprites/dogfight.png")
        cls.spr_logo.set_size(cls.resolution.x / 1920)

        # -------------- Fonts
        cls.font_program = hg.LoadProgramFromAssets("core/shader/font.vsb", "core/shader/font.fsb")
        cls.hud_font = hg.LoadFontFromAssets(cls.hud_font_path, 64)
        cls.title_font = hg.LoadFontFromAssets(cls.title_font_path, 80)
        cls.text_matrx = hg.TransformationMat4(hg.Vec3(0, 0, 0), hg.Vec3(hg.Deg(0), hg.Deg(0), hg.Deg(0)), hg.Vec3(1, -1, 1))
        cls.text_uniform_set_values.push_back(hg.MakeUniformSetValue("u_color", hg.Vec4(1, 1, 0, 1)))
        cls.text_render_state = hg.ComputeRenderState(hg.BM_Alpha, hg.DT_Disabled, hg.FC_Disabled)

        # --------------- Sky & sea render:

        cls.sea_render = PlanetRender(cls.scene, framebuffers_resolution, cls.scene.GetNode("island_clipped").GetTransform().GetPos(), hg.Vec3(-20740.2158, 0, 9793.1535))
        cls.sea_render.load_json_script()

        cls.water_reflexion = WaterReflection(cls.scene, framebuffers_resolution, cls.antialiasing, cls.flag_vr)

        # ---------------- Musics:
        cls.main_music_ref = [hg.LoadWAVSoundAsset("sfx/main_left.wav"), hg.LoadWAVSoundAsset("sfx/main_right.wav")]
        cls.main_music_state = [create_stereo_sound_state(hg.SR_Loop), create_stereo_sound_state(hg.SR_Loop)]

        # --------------- Missions:
        cls.load_json_script()
        Missions.init()

        # ---------------- Physics:
        init_physics(cls.scene, cls.scene_physics, "pictures/height.png", hg.Vec3(cls.sea_render.terrain_position.x, -292, cls.sea_render.terrain_position.z), hg.Vec3(cls.sea_render.terrain_scale.x, 1000, cls.sea_render.terrain_scale.z), hg.Vec2(0, 255))

        cls.scene.Update(0)

    @classmethod
    def update_user_control_mode(cls):
        for machine in cls.destroyables_list:
            user_control_device = machine.get_device("UserControlDevice")
            ia_control_device = machine.get_device("IAControlDevice")
            if user_control_device is not None:
                user_control_device.set_control_mode(cls.control_mode)
            if ia_control_device is not None:
                ia_control_device.set_control_mode(cls.control_mode)

    @classmethod
    def duplicate_scene_lighting(cls, scene_src, scene_dst):
        env0 = scene_src.environment
        env2 = scene_dst.environment
        env2.ambient = env0.ambient
        env2.brdf_map = env0.brdf_map
        env2.fog_color = env0.fog_color
        env2.fog_far = env0.fog_far
        env2.fog_near = env0.fog_near
        env2.irradiance_map = env0.irradiance_map
        env2.radiance_map = env0.radiance_map
        sun0 = scene_src.GetNode("Sun")
        light0 = sun0.GetLight()
        hg.CreateLinearLight(scene_dst, sun0.GetTransform().GetWorld(), light0.GetDiffuseColor(), 1, light0.GetSpecularColor(), 1, light0.GetPriority(), light0.GetShadowType(), 0.008, hg.Vec4(1, 2, 4, 10))

    @classmethod
    def set_activate_sfx(cls, flag):
        if flag != cls.flag_sfx:
            if flag:
                cls.setup_sfx()
            else:
                cls.destroy_sfx()
        cls.flag_sfx = flag

    @classmethod
    def clear_views(cls):
        for vid in range(cls.max_view_id+1):
            hg.SetViewFrameBuffer(vid, hg.InvalidFrameBufferHandle)
            hg.SetViewRect(vid, 0, 0, int(cls.resolution.x), int(cls.resolution.y))
            hg.SetViewClear(vid, hg.CF_Depth, 0x0, 1.0, 0)
            hg.Touch(vid)
        hg.Frame()

    @classmethod
    def set_renderless_mode(cls, flag: bool):
        cls.flag_renderless = flag
        cls.flag_running = False
        if flag:
            cls.flag_activate_particles_mem = Destroyable_Machine.flag_activate_particles
            cls.flag_sfx_mem = cls.flag_sfx
            cls.set_activate_sfx(False)
            Destroyable_Machine.set_activate_particles(False)
            cls.frame_time = 0
        else:
            cls.set_activate_sfx(cls.flag_sfx_mem)
            Destroyable_Machine.set_activate_particles(cls.flag_activate_particles_mem)

        vid = 0
        hg.SetViewFrameBuffer(vid, hg.InvalidFrameBufferHandle)
        hg.SetViewRect(vid, 0, 0, int(cls.resolution.x), int(cls.resolution.y))
        hg.SetViewClear(vid, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
        cls.flag_running = True

    @classmethod
    def update_num_fps(cls, dts):
        cls.nfps[cls.nfps_i] = 1 / dts
        cls.nfps_i = (cls.nfps_i + 1) % len(cls.nfps)
        cls.num_fps = 0
        for ne in cls.nfps:
            cls.num_fps += ne
        cls.num_fps = cls.num_fps / len(cls.nfps)

    @classmethod
    def destroy_players(cls):
        for aircraft in cls.players_ennemies:
            aircraft.destroy()
        for aircraft in cls.players_allies:
            aircraft.destroy()
        for carrier in cls.aircraft_carrier_allies:
            carrier.destroy()
        for carrier in cls.aircraft_carrier_ennemies:
            carrier.destroy()
        for ml in cls.missile_launchers_ennemies:
            ml.destroy()
        for ml in cls.missile_launchers_allies:
            ml.destroy()

        for cockpit in cls.scene_cockpit_aircrafts:
            cockpit.destroy()

        cls.missiles_allies = []
        cls.missiles_ennemies = []
        cls.players_ennemies = []
        cls.players_allies = []
        cls.destroyables_list = []
        cls.aircraft_carrier_allies = []
        cls.aircraft_carrier_ennemies = []
        cls.missile_launchers_allies = []
        cls.missile_launchers_ennemies = []

        cls.scene_cockpit_aircrafts = []


    @classmethod
    def create_aircraft_carriers(cls, num_allies, num_ennemies):

        cls.aircraft_carrier_allies = []
        cls.aircraft_carrier_ennemies = []
        for i in range(num_allies):
            carrier = Carrier("Ally_Carrier_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1)
            cls.aircraft_carrier_allies.append(carrier)
            cls.destroyables_list.append(carrier)
            carrier.add_to_update_list()

        for i in range(num_ennemies):
            carrier = Carrier("Ennemy_Carrier_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2)
            cls.aircraft_carrier_ennemies.append(carrier)
            cls.destroyables_list.append(carrier)
            carrier.add_to_update_list()

    @classmethod
    def create_missiles(cls, machine:Destroyable_Machine, smoke_color):
        md = machine.get_device("MissilesDevice")
        md.set_missiles_config(machine.missiles_config)
        if md is not None:
            for j in range(md.num_slots):
                missile_type = md.missiles_config[j]
                if missile_type == Sidewinder.model_name:
                    missile = Sidewinder(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == Meteor.model_name:
                    missile = Meteor(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == Mica.model_name:
                    missile = Mica(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == AIM_SL.model_name:
                    missile = AIM_SL(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == Karaoke.model_name:
                    missile = Karaoke(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == CFT.model_name:
                    missile = CFT(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)
                if missile_type == S400.model_name:
                    missile = S400(missile_type + machine.name + "." + str(j), cls.scene, cls.scene_physics, cls.pl_resources, machine.nationality)

                md.fit_missile(missile, j)
                missile.set_smoke_color(smoke_color)
            return md.missiles
        return None

    @classmethod
    def create_missile_launchers(cls, num_allies, num_ennemies):
        cls.missile_launchers_allies = []
        cls.missile_launchers_ennemies = []
        cls.num_missile_launchers_allies = num_allies
        cls.num_missile_launchers_ennemies = num_ennemies

        for i in range(num_allies):
            launcher = MissileLauncherS400("Ally_Missile_launcher_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            cls.missile_launchers_allies.append(launcher)
            cls.destroyables_list.append(launcher)
            launcher.add_to_update_list()

            missiles = cls.create_missiles(launcher, cls.allies_missiles_smoke_color)
            if missiles is not None:
                cls.missiles_allies.append([] + missiles)
                cls.destroyables_list += missiles


        for i in range(num_ennemies):
            launcher = MissileLauncherS400("Ennemy_Missile_launcher_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            cls.missile_launchers_ennemies.append(launcher)
            cls.destroyables_list.append(launcher)
            launcher.add_to_update_list()

            missiles = cls.create_missiles(launcher, cls.ennemies_missiles_smoke_color)
            if missiles is not None:
                cls.missiles_ennemies.append([] + missiles)
                cls.destroyables_list += missiles

    @classmethod
    def create_players(cls, allies_types, ennemies_types):
        cls.num_players_allies = len(allies_types)
        cls.num_players_ennemies = len(ennemies_types)
        cls.players_allies = []
        cls.players_ennemies = []
        cls.missiles_allies = []
        cls.missiles_ennemies = []
        cls.players_sfx = []
        cls.missiles_sfx = []


        for i, a_type in enumerate(allies_types):

            if a_type == F14_Parameters.model_name:
                aircraft = F14("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == F14_2_Parameters.model_name:
                aircraft = F14_2("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Rafale_Parameters.model_name:
                aircraft = Rafale("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Eurofighter_Parameters.model_name:
                aircraft = Eurofighter("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == F16_Parameters.model_name:
                aircraft = F16("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == TFX_Parameters.model_name:
                aircraft = TFX("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Miuss_Parameters.model_name:
                aircraft = Miuss("ally_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 1, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))

            cls.destroyables_list.append(aircraft)
            aircraft.add_to_update_list()

            cls.players_allies.append(aircraft)

            missiles = cls.create_missiles(aircraft, cls.allies_missiles_smoke_color)
            if missiles is not None:
                cls.missiles_allies.append([] + missiles)
                cls.destroyables_list += missiles


        for i, a_type in enumerate(ennemies_types):

            if a_type == F14_Parameters.model_name:
                aircraft = F14("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == F14_2_Parameters.model_name:
                aircraft = F14_2("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Rafale_Parameters.model_name:
                aircraft = Rafale("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Eurofighter_Parameters.model_name:
                aircraft = Eurofighter("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == F16_Parameters.model_name:
                aircraft = F16("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == TFX_Parameters.model_name:
                aircraft = TFX("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))
            elif a_type == Miuss_Parameters.model_name:
                aircraft = Miuss("ennemy_" + str(i + 1), cls.scene, cls.scene_physics, cls.pl_resources, 2, hg.Vec3(0, 500, 0), hg.Vec3(0, 0, 0))

            aircraft.add_to_update_list()
            cls.destroyables_list.append(aircraft)

            cls.players_ennemies.append(aircraft)

            missiles = cls.create_missiles(aircraft, cls.ennemies_missiles_smoke_color)
            if missiles is not None:
                cls.missiles_ennemies.append([] + missiles)
                cls.destroyables_list += missiles

        if cls.flag_sfx:
            cls.setup_sfx()

        #cls.scene_physics.SceneCreatePhysicsFromAssets(cls.scene)
        cls.update_user_control_mode()

    @classmethod
    def setup_sfx(cls):
        hg.StopAllSources()
        cls.players_sfx = []
        cls.missiles_sfx = []
        for machine in cls.destroyables_list:
            if machine.type == Destroyable_Machine.TYPE_AIRCRAFT:
                cls.players_sfx.append(AircraftSFX(machine))
            elif machine.type == Destroyable_Machine.TYPE_MISSILE:
                cls.missiles_sfx.append(MissileSFX(machine))

    @classmethod
    def destroy_sfx(cls):
        hg.StopAllSources()
        cls.players_sfx = []
        cls.missiles_sfx = []

    @classmethod
    def init_playground(cls):

        cls.scene.Update(0)

        lt_allies = []
        for carrier in cls.aircraft_carrier_allies:
            lt_allies += carrier.landing_targets
        lt_ennemies = []
        for carrier in cls.aircraft_carrier_ennemies:
            lt_ennemies += carrier.landing_targets

        for i, pl in enumerate(cls.players_allies):
            td = pl.get_device("TargettingDevice")
            td.set_destroyable_targets(cls.players_ennemies)
            pl.set_landing_targets(lt_allies)
            td.targets = cls.players_ennemies
            if cls.num_players_ennemies > 0:
                td.set_target_id(int(uniform(0, 1000) % cls.num_players_ennemies))

        for i, pl in enumerate(cls.missile_launchers_allies):
            td = pl.get_device("TargettingDevice")
            td.set_destroyable_targets(cls.players_ennemies)
            td.targets = cls.players_ennemies
            if cls.num_players_ennemies > 0:
                td.set_target_id(int(uniform(0, 1000) % cls.num_players_ennemies))

        for i, pl in enumerate(cls.players_ennemies):
            td = pl.get_device("TargettingDevice")
            td.set_destroyable_targets(cls.players_allies)
            pl.set_landing_targets(lt_ennemies)
            td.targets = cls.players_allies
            if cls.num_players_allies > 0:
                td.set_target_id(int(uniform(0, 1000) % cls.num_players_allies))

        for i, pl in enumerate(cls.missile_launchers_ennemies):
            td = pl.get_device("TargettingDevice")
            td.set_destroyable_targets(cls.players_allies)
            td.targets = cls.players_allies
            if cls.num_players_allies > 0:
                td.set_target_id(int(uniform(0, 1000) % cls.num_players_allies))


        cls.destroyables_items = {}
        for dm in cls.destroyables_list:
            cls.destroyables_items[dm.name] = dm

        Destroyable_Machine.machines_list = cls.destroyables_list # !!! Move to Destroyable_Machine.__init__()
        Destroyable_Machine.machines_items = cls.destroyables_items # !!! Move to Destroyable_Machine.__init__()

    # ----------------- Views -------------------------------------------------------------------
    @classmethod
    def update_initial_head_matrix(cls, vr_state: hg.OpenVRState):
        mat_head = hg.InverseFast(vr_state.body) * vr_state.head
        rot = hg.GetR(mat_head)
        rot.x = 0
        rot.z = 0
        cls.initial_head_matrix = hg.TransformationMat4(hg.GetT(mat_head), rot)

    @classmethod
    def setup_views_carousel(cls, flag_include_enemies=False):
        if len(cls.aircraft_carrier_allies) > 0:
            fps_start_matrix = cls.aircraft_carrier_allies[0].fps_start_point.GetTransform().GetWorld()
            cls.camera_fps.GetTransform().SetWorld(fps_start_matrix)

        cls.views_carousel = ["fps"]
        for i in range(cls.num_players_allies):
            cls.views_carousel.append("Aircraft_ally_" + str(i + 1))
        for i in range(cls.num_missile_launchers_allies):
            cls.views_carousel.append("MissileLauncher_ally_" + str(i + 1))
        if flag_include_enemies:
            for i in range(cls.num_players_ennemies):
                cls.views_carousel.append("Aircraft_enemy_" + str(i + 1))
            for i in range(cls.num_missile_launchers_ennemies):
                cls.views_carousel.append("MissileLauncher_enemy_" + str(i + 1))
        cls.views_carousel_ptr = 1

    @classmethod
    def get_player_from_caroursel_id(cls, view_id=""):
        if view_id == "":
            view_id = cls.views_carousel[cls.views_carousel_ptr]
        if view_id == "fps":
            return None

        spl = view_id.split("_")
        machine_type, nation, num = spl[0], spl[1], spl[2]
        m_id = int(num) - 1

        if machine_type == "Aircraft":
            if nation == "ally":
                return cls.players_allies[m_id]
            elif nation == "enemy":
                return cls.players_ennemies[m_id]
            else:
                return None
        if machine_type == "MissileLauncher":
            if nation == "ally":
                return cls.missile_launchers_allies[m_id]
            elif nation == "enemy":
                return cls.missile_launchers_ennemies[m_id]
            else:
                return None

    @classmethod
    def set_view_carousel(cls, view_id):
        if view_id == "fps":
            cls.views_carousel_ptr = 0
            cls.update_main_view_from_carousel()
        else:
            for i in range(len(cls.views_carousel)):
                if cls.views_carousel[i] == view_id:
                    cls.views_carousel_ptr = i
                    cls.update_main_view_from_carousel()

    @classmethod
    def update_main_view_from_carousel(cls):
        view_id = cls.views_carousel[cls.views_carousel_ptr]
        if view_id == "fps":
            cls.smart_camera.setup(SmartCamera.TYPE_FPS, cls.camera_fps)
            cls.scene.SetCurrentCamera(cls.camera_fps)
        else:
            player = cls.get_player_from_caroursel_id(view_id)
            cls.smart_camera.set_camera_tracking_target_distance(player.camera_track_distance)
            cls.smart_camera.set_camera_follow_distance(player.camera_follow_distance)
            cls.smart_camera.set_tactical_camera_distance(player.camera_tactical_distance)
            cls.smart_camera.set_tactical_min_altitude(player.camera_tactical_min_altitude)
            if cls.player_view_mode == SmartCamera.TYPE_TACTICAL:
                camera = cls.camera
                td = player.get_device("TargettingDevice")
                target = td.get_target()
                if target is not None:
                    target = target.get_parent_node()
                cls.smart_camera.setup_tactical(camera, player.get_parent_node(), target, None)

            else:
                if cls.player_view_mode == SmartCamera.TYPE_FIX:
                    target_node = player.get_current_pilot_head()
                    camera = cls.camera_cokpit
                else:
                    target_node = player.get_parent_node()
                    camera = cls.camera
                cls.smart_camera.setup(cls.player_view_mode, camera, target_node)

            cls.scene.SetCurrentCamera(camera)

    @classmethod
    def set_track_view(cls, view_name):
        if cls.satellite_view:
            cls.satellite_view = False
            cls.update_main_view_from_carousel()
        cls.smart_camera.set_track_view(view_name)
        

    @classmethod
    def activate_cockpit_view(cls):
        if not cls.flag_cockpit_view:
            if cls.user_aircraft is not None:
                if cls.user_aircraft.get_current_pilot_head() is not None:
                    cls.flag_cockpit_view = True
                    cls.player_view_mode = SmartCamera.TYPE_FIX

    @classmethod
    def deactivate_cockpit_view(cls):
        if cls.flag_cockpit_view:
            cls.player_view_mode = SmartCamera.TYPE_TRACKING
            cls.set_track_view("back")
            cls.flag_cockpit_view = False


    @classmethod
    def switch_cockpit_view(cls, new_user_aircraft: Aircraft):
        if cls.flag_cockpit_view:
            if new_user_aircraft.get_current_pilot_head() is None:
                cls.deactivate_cockpit_view()


    @classmethod
    def control_views(cls, keyboard):
        quit_sv = False
        if keyboard.Down(hg.K_Numpad2):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_TRACKING
            cls.update_main_view_from_carousel()
            cls.set_track_view("back")
        elif keyboard.Down(hg.K_Numpad8):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_TRACKING
            cls.update_main_view_from_carousel()
            cls.set_track_view("front")
        elif keyboard.Down(hg.K_Numpad4):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_TRACKING
            cls.update_main_view_from_carousel()
            cls.set_track_view("left")
        elif keyboard.Down(hg.K_Numpad6):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_TRACKING
            cls.update_main_view_from_carousel()
            cls.set_track_view("right")

        elif keyboard.Pressed(hg.K_Numpad5):
            tgt = cls.get_player_from_caroursel_id()
            if tgt is not None:
                cls.deactivate_cockpit_view()
                tgt = tgt.get_parent_node()
                if not cls.satellite_view:
                    cls.satellite_view = True
                    cls.smart_camera.setup(SmartCamera.TYPE_SATELLITE, cls.satellite_camera, tgt)
                    cls.scene.SetCurrentCamera(cls.satellite_camera)

        elif keyboard.Pressed(hg.K_Numpad1):
            if not cls.flag_network_mode:
                if cls.user_aircraft is not None:
                    uctrl = cls.user_aircraft.get_device("UserControlDevice")
                    if uctrl is not None:
                        uctrl.deactivate()
                    #ia = cls.user_aircraft.get_device("IAControlDevice")
                    #if ia is not None:
                    #    ia.activate()
            cls.views_carousel_ptr += 1
            if cls.views_carousel_ptr >= len(cls.views_carousel):
                cls.views_carousel_ptr = 0
            if cls.views_carousel[cls.views_carousel_ptr] != "fps":
                new_user_aircraft = cls.get_player_from_caroursel_id(cls.views_carousel[cls.views_carousel_ptr])
                cls.switch_cockpit_view(new_user_aircraft)
                cls.user_aircraft = new_user_aircraft
                cls.user_aircraft.set_focus()
                ia = cls.user_aircraft.get_device("IAControlDevice")
                if ia is None:
                    uctrl = cls.user_aircraft.get_device("UserControlDevice")
                    if uctrl is not None:
                        uctrl.activate()
                elif not ia.is_activated():
                    apctrl = cls.user_aircraft.get_device("AutopilotControlDevice")
                    if apctrl is not None:
                        apctrl.deactivate()
                    uctrl = cls.user_aircraft.get_device("UserControlDevice")
                    if uctrl is not None:
                        uctrl.activate()
                    else:
                        ia.activate()
                # cls.user_aircraft.deactivate_IA()
            else:
                cls.deactivate_cockpit_view()
                cls.satellite_view = False
                cls.user_aircraft = None
            cls.update_main_view_from_carousel()

        elif keyboard.Pressed(hg.K_Numpad3):
            quit_sv = True
            cls.activate_cockpit_view()
            cls.update_main_view_from_carousel()

        elif keyboard.Pressed(hg.K_Numpad9):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_FOLLOW
            cls.update_main_view_from_carousel()

        elif keyboard.Pressed(hg.K_Numpad7):
            cls.deactivate_cockpit_view()
            quit_sv = True
            cls.player_view_mode = SmartCamera.TYPE_TACTICAL
            cls.update_main_view_from_carousel()

        if quit_sv and cls.satellite_view:
            cls.satellite_view = False
            if cls.player_view_mode == SmartCamera.TYPE_FOLLOW:
                camera = cls.camera
            elif cls.player_view_mode == SmartCamera.TYPE_TRACKING:
                camera = cls.camera
            elif cls.player_view_mode == SmartCamera.TYPE_FIX:
                camera = cls.camera_cokpit
            elif cls.player_view_mode == SmartCamera.TYPE_TACTICAL:
                camera = cls.camera
            cls.scene.SetCurrentCamera(camera)

        if cls.satellite_view:
            if keyboard.Down(hg.K_Insert):
                cls.smart_camera.increment_satellite_view_size()
            elif keyboard.Down(hg.K_PageUp):
                cls.smart_camera.decrement_satellite_view_size()
        else:
            if keyboard.Down(hg.K_Insert):
                cls.scene.GetCurrentCamera().GetCamera().SetFov(cls.scene.GetCurrentCamera().GetCamera().GetFov() * 0.99)
            elif keyboard.Down(hg.K_PageUp):
                cls.scene.GetCurrentCamera().GetCamera().SetFov(cls.scene.GetCurrentCamera().GetCamera().GetFov() * 1.01)


    # =============================== Scene datas
    @classmethod
    def get_current_camera(cls):
        return cls.scene.GetCurrentCamera()

    # =============================== Displays =============================================

    @classmethod
    def display_landing_trajectory(cls, landing_target: LandingTarget):
        if landing_target is not None:
            num_steps = 100
            c = hg.Color(0, 1, 0, 1)
            p0 = landing_target.get_position(0)
            step = landing_target.horizontal_amplitude / num_steps * 2
            for i in range(1, num_steps):
                p1 = landing_target.get_position(step * i)
                Overlays.add_line(p0, p1, c, c)
                p0 = p1

    @classmethod
    def display_landing_projection(cls, aircraft: Aircraft):
        ia_ctrl = aircraft.get_device("IAControlDevice")
        if ia_ctrl is not None and ia_ctrl.IA_landing_target is not None:
            c = hg.Color(1, 0, 0, 1)
            c2 = hg.Color(0, 1, 0, 1)
            p = ia_ctrl.calculate_landing_projection(aircraft, ia_ctrl.IA_landing_target)
            target_point = ia_ctrl.calculate_landing_target_point(aircraft, ia_ctrl.IA_landing_target, p)
            if p is not None:
                v = ia_ctrl.IA_landing_target.get_landing_vector()
                vb = hg.Vec3(v.y, 0, -v.x)
                Overlays.add_line(p - vb * 50, p + vb * 50, c, c)
                Overlays.add_line(target_point, aircraft.parent_node.GetTransform().GetPos(), c2, c)

    @classmethod
    def display_machine_vectors(cls, machine: Destroyable_Machine):
        pos = machine.get_position()
        if machine.flag_display_linear_speed:
            display_vector(pos, machine.get_move_vector(), "linear speed", hg.Vec2(0, 0.03), hg.Color.Yellow)
        hs, vs = machine.get_world_speed()
        if machine.flag_display_vertical_speed:
            display_vector(pos, hg.Vec3.Up * vs, "Vertical speed", hg.Vec2(0, 0.02), hg.Color.Red)
        if machine.flag_display_horizontal_speed:
            az = machine.get_Z_axis()
            ah = hg.Normalize(hg.Vec3(az.x, 0, az.z))
            display_vector(pos, ah * hs, "Horizontal speed",hg.Vec2(0, 0.01), hg.Color.Green)

    # =============================== 2D HUD =============================================

    @classmethod
    def get_2d_hud(cls, point3d: hg.Vec3):
        if cls.flag_vr:
            cam = cls.scene.GetCurrentCamera()
            fov = atan(cls.vr_hud.y / (2 * cls.vr_hud.z)) * 2
            main_camera_matrix = cam.GetTransform().GetWorld()
            vs = hg.ComputePerspectiveViewState(main_camera_matrix, fov, cam.GetCamera().GetZNear(), cam.GetCamera().GetZFar(), hg.Vec2(cls.vr_hud.x / cls.vr_hud.y, 1))
            pos_view = vs.view * point3d
        else:
            vs = cls.scene.ComputeCurrentCameraViewState(hg.Vec2(cls.resolution.x / cls.resolution.y, 1))
            pos_view = vs.view * point3d
        f, pos2d = hg.ProjectToScreenSpace(vs.proj, pos_view, cls.resolution)
        if f:
            return hg.Vec2(pos2d.x, pos2d.y)
        else:
            return None

    # =============================== GUI =============================================
    @classmethod
    def update_missiles_smoke_color(cls):
        for missile_t in cls.missiles_allies:
            for missile in missile_t:
                missile.set_smoke_color(cls.allies_missiles_smoke_color)
        for missile_t in cls.missiles_ennemies:
            for missile in missile_t:
                missile.set_smoke_color(cls.ennemies_missiles_smoke_color)
    
    @classmethod
    def gui(cls):
        aircrafts = cls.players_allies + cls.players_ennemies

        if hg.ImGuiBegin("Main Settings"):

            hg.ImGuiSetWindowPos("Main Settings",hg.Vec2(10, 60), hg.ImGuiCond_Once)
            hg.ImGuiSetWindowSize("Main Settings",hg.Vec2(650,625), hg.ImGuiCond_Once)

            if hg.ImGuiButton("Load simulator parameters"):
                cls.load_json_script()
                cls.update_missiles_smoke_color()
            hg.ImGuiSameLine()
            if hg.ImGuiButton("Save simulator parameters"):
                cls.save_json_script()

            hg.ImGuiText("Num nodes: %d" % cls.scene.GetNodeCount())

            d, f = hg.ImGuiCheckbox("Display FPS", cls.flag_display_fps)
            if d: cls.flag_display_fps = f
            d, f = hg.ImGuiCheckbox("Display HUD", cls.flag_display_HUD)
            if d: cls.flag_display_HUD = f

            d, f = hg.ImGuiCheckbox("Renderless", cls.flag_renderless)
            if d: cls.set_renderless_mode(f)
            d, f = hg.ImGuiCheckbox("Display radar in renderless mode", cls.flag_display_radar_in_renderless)
            if d: cls.flag_display_radar_in_renderless = f
            d, f = hg.ImGuiCheckbox("Control views", cls.flag_control_views)
            if d: cls.flag_control_views = f
            d, f = hg.ImGuiCheckbox("Particles", Destroyable_Machine.flag_activate_particles)
            if d: Destroyable_Machine.set_activate_particles(f)
            d, f = hg.ImGuiCheckbox("SFX", cls.flag_sfx)
            if d: cls.set_activate_sfx(f)

            d, f = hg.ImGuiCheckbox("Display landing trajectories", cls.flag_display_landing_trajectories)
            if d: cls.flag_display_landing_trajectories = f

            d, f = hg.ImGuiCheckbox("Display machines bounds", cls.flag_display_machines_bounding_boxes)
            if d: cls.flag_display_machines_bounding_boxes = f

            d, f = hg.ImGuiCheckbox("Display physics debug", cls.flag_display_physics_debug)
            if d: cls.flag_display_physics_debug = f

            f, c = hg.ImGuiColorEdit("Allies missiles smoke color", cls.allies_missiles_smoke_color)
            if f:
                cls.allies_missiles_smoke_color = c
                cls.update_missiles_smoke_color()
            f, c = hg.ImGuiColorEdit("Ennmies missiles smoke color", cls.ennemies_missiles_smoke_color)
            if f:
                cls.ennemies_missiles_smoke_color = c
                cls.update_missiles_smoke_color()

            # Aircrafts:
            d, f = hg.ImGuiCheckbox("Display selected aircraft", cls.flag_display_selected_aircraft)
            if d: cls.flag_display_selected_aircraft = f

            aircrafts_list = hg.StringList()

            for aircraft in aircrafts:
                nm = aircraft.name
                if aircraft == cls.user_aircraft:
                    nm += " - USER -"
                aircrafts_list.push_back(nm)

            f, d = hg.ImGuiListBox("Aircrafts", cls.selected_aircraft_id, aircrafts_list,20)
            if f:
                cls.selected_aircraft_id = d

        hg.ImGuiEnd()

        cls.selected_aircraft = aircrafts[cls.selected_aircraft_id]
        cls.selected_aircraft.gui()


    @classmethod
    def load_json_script(cls, file_name="scripts/simulator_parameters.json"):
        file = hg.OpenText(file_name)
        if not file:
            print("ERROR - Can't open json file : " + file_name)
        else:
            json_script = hg.ReadString(file)
            hg.Close(file)
            if json_script != "":
                script_parameters = json.loads(json_script)
                cls.allies_missiles_smoke_color = list_to_color(script_parameters["allies_missiles_smoke_color"])
                cls.ennemies_missiles_smoke_color = list_to_color(script_parameters["ennemies_missiles_smoke_color"])

    @classmethod
    def save_json_script(cls, output_filename="scripts/simulator_parameters.json"):
        script_parameters = {"allies_missiles_smoke_color": color_to_list(cls.allies_missiles_smoke_color),
                             "ennemies_missiles_smoke_color": color_to_list(cls.ennemies_missiles_smoke_color),
                             }
        json_script = json.dumps(script_parameters, indent=4)
        file = hg.OpenWrite(output_filename)
        if file:
            hg.WriteString(file, json_script)
            hg.Close(file)
            return True
        else:
            print("ERROR - Can't open json file : " + output_filename)
            return False

    # ================================ Scene update and rendering modes ============================================

    @classmethod
    def update_kinetics(cls, dts):
        #for dm in Destroyable_Machine.update_list:
        #    dm.update_collision_nodes_matrices()

        for dm in Destroyable_Machine.update_list:
            dm.update_kinetics(dts)
            cls.display_machine_vectors(dm)


    @classmethod
    def clear_display_lists(cls):
        cls.sprites_display_list = []
        #cls.texts_display_list = []
        Overlays.texts2D_display_list = []
        Overlays.texts3D_display_list = []
        Overlays.lines = []

    @classmethod
    def render_frame_vr(cls):

        vid = 0
        views = hg.SceneForwardPipelinePassViewId()

        camera = cls.scene.GetCurrentCamera()
        main_camera_matrix = camera.GetTransform().GetWorld()
        body_mtx = main_camera_matrix * hg.InverseFast(cls.initial_head_matrix)

        cls.vr_state = hg.OpenVRGetState(body_mtx, camera.GetCamera().GetZNear(), camera.GetCamera().GetZFar())
        vs_left, vs_right = hg.OpenVRStateToViewState(cls.vr_state)

        vr_eye_rect = hg.IntRect(0, 0, int(cls.vr_state.width), int(cls.vr_state.height))

        # ========== Display Reflect scene ===================

        # Deactivated because assymetric VR FOV not resolved.
        """
        cls.scene.canvas.color = hg.Color(1, 0, 0, 1)  # En attendant de fixer le pb de la depth texture du framebuffer.

        cls.scene.canvas.clear_z = True
        cls.scene.canvas.clear_color = True
        left_reflect, right_reflect = cls.water_reflexion.compute_vr_reflect(camera, cls.vr_state, vs_left, vs_right)

        vid, passId = hg.PrepareSceneForwardPipelineCommonRenderData(vid, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)

        # Prepare the left eye render data then draw to its framebuffer
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, left_reflect, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, vr_eye_rect, left_reflect, cls.pipeline, cls.render_data, cls.pl_resources, cls.water_reflexion.quad_frameBuffer_left.handle)

        # Prepare the right eye render data then draw to its framebuffer
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, right_reflect, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, vr_eye_rect, right_reflect, cls.pipeline, cls.render_data, cls.pl_resources, cls.water_reflexion.quad_frameBuffer_right.handle)
        """
    
        # ========== Display raymarch scene ===================
        output_fb_left = cls.vr_left_fb #cls.post_process.quad_frameBuffer_left
        output_fb_right = cls.vr_right_fb #cls.post_process.quad_frameBuffer_right
        cls.scene.canvas.clear_z = True
        cls.scene.canvas.clear_color = True

        #tex_reflect_left_color = hg.GetColorTexture(cls.water_reflexion.quad_frameBuffer_left)
        #tex_reflect_left_depth = hg.GetDepthTexture(cls.water_reflexion.quad_frameBuffer_left)
        #tex_reflect_right_color = hg.GetColorTexture(cls.water_reflexion.quad_frameBuffer_right)
        #tex_reflect_right_depth = hg.GetDepthTexture(cls.water_reflexion.quad_frameBuffer_right)
        vid = cls.sea_render.render_vr(vid, cls.vr_state, vs_left, vs_right, output_fb_left, output_fb_right) #, tex_reflect_left_color, tex_reflect_left_depth, tex_reflect_right_color, tex_reflect_right_depth)


        # ========== Display models scene =======================
        cls.scene.canvas.clear_z = False
        cls.scene.canvas.clear_color = False
        vid, passId = hg.PrepareSceneForwardPipelineCommonRenderData(vid, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)

        # Prepare the left eye render data then draw to its framebuffer
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, vs_left, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, vr_eye_rect, vs_left, cls.pipeline, cls.render_data, cls.pl_resources, output_fb_left.GetHandle())

        # Prepare the right eye render data then draw to its framebuffer
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, vs_right, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, vr_eye_rect, vs_right, cls.pipeline, cls.render_data, cls.pl_resources, output_fb_right.GetHandle())

        # ==================== Display 3D Overlays ===========

        #Overlays.add_text3D("HELLO WORLD", hg.Vec3(0, 50, 200), 1, hg.Color.Red)

        if len(Overlays.texts3D_display_list) > 0 or len(Overlays.lines) > 0:

            #hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer_left.handle)
            hg.SetViewFrameBuffer(vid, output_fb_left.GetHandle())
            hg.SetViewRect(vid, 0, 0, int(cls.vr_state.width), int(cls.vr_state.height))
            hg.SetViewClear(vid, hg.CF_Depth, 0, 1.0, 0)
            hg.SetViewTransform(vid, vs_left.view, vs_left.proj)
            eye_left = cls.vr_state.head * cls.vr_state.left.offset
            Overlays.display_texts3D(vid, eye_left)
            Overlays.draw_lines(vid)
            vid += 1

            #hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer_right.handle)
            hg.SetViewFrameBuffer(vid, output_fb_right.GetHandle())
            hg.SetViewRect(vid, 0, 0, int(cls.vr_state.width), int(cls.vr_state.height))
            hg.SetViewClear(vid, hg.CF_Depth, 0, 1.0, 0)
            hg.SetViewTransform(vid, cls.vr_viewstate.vs_right.view, cls.vr_viewstate.vs_right.proj)
            eye_right = cls.vr_state.head * cls.vr_state.right.offset
            Overlays.display_texts3D(vid, eye_right)
            Overlays.draw_lines(vid)
            vid += 1


        # ==================== Display 2D sprites ===========

        cam_mat = cls.scene.GetCurrentCamera().GetTransform().GetWorld()
        mat_spr = cam_mat  # * vr_state.initial_head_offset

        #hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer_left.handle)
        hg.SetViewFrameBuffer(vid, output_fb_left.GetHandle())
        hg.SetViewRect(vid, 0, 0, int(cls.vr_state.width), int(cls.vr_state.height))
        hg.SetViewClear(vid, hg.CF_Depth, 0, 1.0, 0)
        hg.SetViewTransform(vid, vs_left.view, vs_left.proj)

        """
        for txt in cls.texts_display_list:
            if "h_align" in txt:
                cls.display_text_vr(vid, mat_spr, cls.resolution, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"], txt["h_align"])
            else:
                cls.display_text_vr(vid, mat_spr, cls.resolution, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"])
        """

        z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_left.proj)

        Overlays.display_texts2D_vr(vid, cls.initial_head_matrix, z_near, z_far, cls.resolution, mat_spr, cls.vr_hud)

        for spr in cls.sprites_display_list:
            spr.draw_vr(vid, mat_spr, cls.resolution, cls.vr_hud)
        vid += 1

        #hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer_right.handle)
        hg.SetViewFrameBuffer(vid, output_fb_right.GetHandle())
        hg.SetViewRect(vid, 0, 0, int(cls.vr_state.width), int(cls.vr_state.height))
        hg.SetViewClear(vid, hg.CF_Depth, 0, 1.0, 0)
        hg.SetViewTransform(vid, vs_right.view, vs_right.proj)

        """
        for txt in cls.texts_display_list:
            if "h_align" in txt:
                cls.display_text_vr(vid, mat_spr, cls.resolution, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"], txt["h_align"])
            else:
                cls.display_text_vr(vid, mat_spr, cls.resolution, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"])
        """
        z_near, z_far = hg.ExtractZRangeFromProjectionMatrix(vs_right.proj)
        Overlays.display_texts2D_vr(vid, cls.initial_head_matrix, z_near, z_far, cls.resolution, mat_spr, cls.vr_hud)

        for spr in cls.sprites_display_list:
            spr.draw_vr(vid, mat_spr, cls.resolution, cls.vr_hud)
        vid += 1

        # ============= Post-process

        #vid = cls.post_process.display_vr(vid, cls.vr_state, vs_left, vs_right, cls.vr_left_fb, cls.vr_right_fb, cls.pl_resources)

        # ============= Display the VR eyes texture to the backbuffer =============
        hg.SetViewRect(vid, 0, 0, int(cls.resolution.x), int(cls.resolution.y))
        hg.SetViewClear(vid, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
        vs = hg.ComputeOrthographicViewState(hg.TranslationMat4(hg.Vec3(0, 0, 0)), cls.resolution.y, 0.1, 100, hg.ComputeAspectRatioX(cls.resolution.x, cls.resolution.y))
        hg.SetViewTransform(vid, vs.view, vs.proj)

        cls.vr_quad_uniform_set_texture_list.clear()
        #cls.vr_quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.OpenVRGetColorTexture(cls.vr_left_fb), 0))
        cls.vr_quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.GetColorTexture(cls.post_process.quad_frameBuffer_left), 0))
        hg.SetT(cls.vr_quad_matrix, hg.Vec3(cls.eye_t_x, 0, 1))
        hg.DrawModel(vid, cls.vr_quad_model, cls.vr_tex0_program, cls.vr_quad_uniform_set_value_list, cls.vr_quad_uniform_set_texture_list, cls.vr_quad_matrix, cls.vr_quad_render_state)

        cls.vr_quad_uniform_set_texture_list.clear()
        #cls.vr_quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.OpenVRGetColorTexture(cls.vr_right_fb), 0))
        cls.vr_quad_uniform_set_texture_list.push_back(hg.MakeUniformSetTexture("s_tex", hg.GetColorTexture(cls.post_process.quad_frameBuffer_right), 0))
        hg.SetT(cls.vr_quad_matrix, hg.Vec3(-cls.eye_t_x, 0, 1))
        hg.DrawModel(vid, cls.vr_quad_model, cls.vr_tex0_program, cls.vr_quad_uniform_set_value_list, cls.vr_quad_uniform_set_texture_list, cls.vr_quad_matrix, cls.vr_quad_render_state)


    @classmethod
    def render_frame(cls):
        vid = 0
        views = hg.SceneForwardPipelinePassViewId()
        res_x = int(cls.resolution.x)
        res_y = int(cls.resolution.y)
        # ========== Display Reflect scene ===================
        cls.water_reflexion.set_camera(cls.scene)

        #cls.scene.canvas.color = cls.sea_render.high_atmosphere_color
        cls.scene.canvas.color = hg.Color(1, 0, 0, 1) # En attendant de fixer le pb de la depth texture du framebuffer.

        cls.scene.canvas.clear_z = True
        cls.scene.canvas.clear_color = True
        # hg.SetViewClear(vid, 0, 0x0, 1.0, 0)

        vs = cls.scene.ComputeCurrentCameraViewState(hg.ComputeAspectRatioX(res_x, res_y))
        vid, passId = hg.PrepareSceneForwardPipelineCommonRenderData(vid, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, vs, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)

        # Get quad_frameBuffer.handle to define output frameBuffer
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, hg.IntRect(0, 0, res_x, res_y), vs, cls.pipeline, cls.render_data, cls.pl_resources, cls.water_reflexion.quad_frameBuffer.handle)

        cls.water_reflexion.restore_camera(cls.scene)

        # ========== Display raymarch scene ===================

        cls.scene.canvas.clear_z = True
        cls.scene.canvas.clear_color = True
        c = cls.scene.GetCurrentCamera()
        vid = cls.sea_render.render(vid, c, hg.Vec2(res_x, res_y), hg.GetColorTexture(cls.water_reflexion.quad_frameBuffer), hg.GetDepthTexture(cls.water_reflexion.quad_frameBuffer), cls.post_process.quad_frameBuffer)

        # ========== Display models scene ===================

        cls.scene.canvas.clear_z = False
        cls.scene.canvas.clear_color = False
        # hg.SetViewClear(vid, hg.ClearColor | hg.ClearDepth, 0x0, 1.0, 0)

        vs = cls.scene.ComputeCurrentCameraViewState(hg.ComputeAspectRatioX(res_x, res_y))
        vid, passId = hg.PrepareSceneForwardPipelineCommonRenderData(vid, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)
        vid, passId = hg.PrepareSceneForwardPipelineViewDependentRenderData(vid, vs, cls.scene, cls.render_data, cls.pipeline, cls.pl_resources, views)

        # Get quad_frameBuffer.handle to define output frameBuffer
        vid, passId = hg.SubmitSceneToForwardPipeline(vid, cls.scene, hg.IntRect(0, 0, res_x, res_y), vs, cls.pipeline, cls.render_data, cls.pl_resources, cls.post_process.quad_frameBuffer.handle)

        # ==================== Display 3D Overlays ===========
        hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer.handle)
        hg.SetViewRect(vid, 0, 0, res_x, res_y)
        cam = cls.scene.GetCurrentCamera()
        hg.SetViewClear(vid, hg.CF_Depth, 0, 1.0, 0)
        cam_mat = cam.GetTransform().GetWorld()
        view_matrix = hg.InverseFast(cam_mat)
        c = cam.GetCamera()
        projection_matrix = hg.ComputePerspectiveProjectionMatrix(c.GetZNear(), c.GetZFar(), hg.FovToZoomFactor(c.GetFov()), hg.Vec2(res_x / res_y, 1))
        hg.SetViewTransform(vid, view_matrix, projection_matrix)

        #Overlays.add_text3D("HELLO WORLD", hg.Vec3(0, 50, 200), 1, hg.Color.Red)

        Overlays.display_texts3D(vid, cls.scene.GetCurrentCamera().GetTransform().GetWorld())
        Overlays.draw_lines(vid)
        if cls.flag_display_physics_debug:
            Overlays.display_physics_debug(vid, cls.scene_physics)

        vid += 1
        # ==================== Display 2D sprites ===========

        hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer.handle)
        hg.SetViewRect(vid, 0, 0, res_x, res_y)

        Sprite.setup_matrix_sprites2D(vid, cls.resolution)

        for spr in cls.sprites_display_list:
            spr.draw(vid)

        vid += 1

        # ==================== Display 2D texts ===========

        hg.SetViewFrameBuffer(vid, cls.post_process.quad_frameBuffer.handle)
        hg.SetViewRect(vid, 0, 0, res_x, res_y)

        Sprite.setup_matrix_sprites2D(vid, cls.resolution)

        """
        for txt in cls.texts_display_list:
            if "h_align" in txt:
                cls.display_text(vid, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"], txt["h_align"])
            else:
                cls.display_text(vid, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"])
        """

        #Overlays.add_text2D_from_3D_position("HELLO World !", hg.Vec3(0, 50, 200), hg.Vec2(-0.1, 0), 0.02, hg.Color.Red)
        Overlays.display_texts2D(vid, cls.scene.GetCurrentCamera(), cls.resolution)


        vid += 1
        # ========== Post process:
        cls.scene.canvas.clear_z = True
        cls.scene.canvas.clear_color = True
        cls.post_process.display(vid, cls.pl_resources, cls.resolution)
        cls.max_view_id = vid


    @classmethod
    def update_renderless(cls, dt):
        vid = 0
        res_x = int(cls.resolution.x)
        res_y = int(cls.resolution.y)
        cls.frame_time += dt
        if hg.time_to_sec_f(cls.frame_time) >= 1 / 60:
            hg.SetViewRect(0, 0, 0, res_x, res_y)
            hg.SetViewClear(0, hg.CF_Color | hg.CF_Depth, 0x0, 1.0, 0)
            Sprite.setup_matrix_sprites2D(vid, cls.resolution)
            for spr in cls.sprites_display_list:
                spr.draw(vid)
            #cls.texts_display_list.append({"text": "RENDERLESS MODE", "font": cls.hud_font, "pos": hg.Vec2(0.5, 0.5), "size": 0.018, "color": hg.Color.Red})
            Overlays.add_text2D("RENDERLESS MODE", hg.Vec2(0.5, 0.5), 0.018, hg.Color.Red, cls.hud_font)
            """
            for txt in cls.texts_display_list:
                if "h_align" in txt:
                    cls.display_text(vid, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"], txt["h_align"])
                else:
                    cls.display_text(vid, txt["text"], txt["pos"], txt["size"], txt["font"], txt["color"])
            """
            Overlays.display_texts2D(vid, cls.scene.GetCurrentCamera(), cls.resolution)

            if cls.flag_gui:
                hg.ImGuiEndFrame(255)
            hg.Frame()
            hg.UpdateWindow(cls.win)
            cls.frame_time = 0

    @classmethod
    def update_inputs(cls):
        if cls.flag_running:
            cls.keyboard.Update()
            cls.mouse.Update()

            if cls.gamepad is not None:
                cls.gamepad.Update()
                if cls.gamepad.IsConnected():
                    cls.flag_paddle = True
                else:
                    cls.flag_paddle = False
            else:
                cls.flag_paddle = False

            if cls.generic_controller is not None:
                cls.generic_controller.Update()
                if cls.generic_controller.IsConnected():
                    cls.flag_generic_controller = True
                else:
                    cls.flag_generic_controller = False
            else:
                cls.flag_generic_controller = False

    @classmethod
    def client_update(cls):
        cls.flag_client_ask_update_scene = True

    @classmethod
    def update(cls):
        if cls.flag_running:
            #cls.t = hg.time_to_sec_f(hg.GetClock())
            #cls.update_inputs()

            real_dt = hg.TickClock()
            forced_dt = hg.time_from_sec_f(cls.timestep)

            if cls.keyboard.Pressed(hg.K_Escape):
                cls.flag_exit = True
            if get_connected()[0]:
                if get_button_values(get_state(0))['BACK'] and get_button_values(get_state(0))['START']:
                    cls.flag_exit = True
            if cls.flag_vr:
                if cls.vr_state is not None and cls.keyboard.Pressed(hg.K_F11):
                    cls.update_initial_head_matrix(cls.vr_state)

            if cls.keyboard.Pressed(hg.K_F12):
                cls.flag_gui = not cls.flag_gui

            if cls.keyboard.Pressed(hg.K_F10):
                cls.flag_display_HUD = not cls.flag_display_HUD

            if cls.flag_gui:
                hg.ImGuiBeginFrame(int(cls.resolution.x), int(cls.resolution.y), real_dt, hg.ReadMouse(), hg.ReadKeyboard())
                cls.smart_camera.update_hovering_ImGui()
                cls.gui()
                cls.sea_render.gui(cls.scene.GetCurrentCamera().GetTransform().GetPos())
                ParticlesEngine.gui()

            if cls.flag_display_fps:
                cls.update_num_fps(hg.time_to_sec_f(real_dt))
                #cls.texts_display_list.append({"text": "FPS %d" % (cls.num_fps), "font": cls.hud_font, "pos": hg.Vec2(0.001, 0.999), "size": 0.018, "color": hg.Color.Yellow})
                Overlays.add_text2D("FPS %d" % (cls.num_fps), hg.Vec2(0.001, 0.999), 0.018, hg.Color.Yellow, cls.hud_font)

            # =========== State update:
            if cls.flag_renderless:
                used_dt = forced_dt
            else:
                used_dt = min(forced_dt * 2, real_dt)
            cls.current_state = cls.current_state(hg.time_to_sec_f(used_dt)) # Minimum frame rate security
            hg.SceneUpdateSystems(cls.scene, cls.clocks, used_dt, cls.scene_physics, used_dt, 1000)  # ,10,1000)

            # =========== Render scene visuals:
            if not cls.flag_renderless:

                if cls.flag_vr:
                    cls.render_frame_vr()
                else:
                    cls.render_frame()

                if cls.flag_gui:
                    hg.ImGuiEndFrame(255)
                hg.Frame()
                if cls.flag_vr:
                    hg.OpenVRSubmitFrame(cls.vr_left_fb, cls.vr_right_fb)
                #hg.UpdateWindow(cls.win)

            # =========== Renderless mode:
            else:
                cls.update_renderless(forced_dt)

            cls.clear_display_lists()

        cls.flag_client_ask_update_scene = False

    @classmethod
    def update_window(cls):
        #if not cls.flag_renderless_mode:
        hg.UpdateWindow(cls.win)

    # ================================ Network ============================================

    @classmethod
    def get_network(cls):
        return get_network()



def init_menu_phase():
    Main.flag_running = False
    Main.set_renderless_mode(False)
    Main.flag_network_mode = False
    Main.fading_to_next_state = False
    Main.destroy_sfx()
    ParticlesEngine.reset_engines()
    Destroyable_Machine.reset_machines()
    Main.create_aircraft_carriers(1, 0)
    Missions.setup_carriers(Main.aircraft_carrier_allies, hg.Vec3(0, 0, 0), hg.Vec3(100, 0, 25), 0)
    Main.create_players(["Rafale", "Eurofighter", "Rafale", "Eurofighter", "Rafale", "F16"], [])
    n = len(Main.players_allies)
    Missions.aircrafts_starts_on_carrier(Main.players_allies[0:n // 2], Main.aircraft_carrier_allies[0], hg.Vec3(10, 19.5, 60), 0, hg.Vec3(0, 0, -20))
    Missions.aircrafts_starts_on_carrier(Main.players_allies[n // 2:n], Main.aircraft_carrier_allies[0], hg.Vec3(-10, 19.5, 40), 0, hg.Vec3(0, 0, -20))
    nd = Main.scene.GetNode("aircraft_carrier")

    if Main.intro_anim_id == 1:
        Main.anim_camera_intro_dist = Animation(5, 30, 80, 250)
        Main.anim_camera_intro_rot = Animation(0, 20, 1, 20)
        Main.camera_intro.GetTransform().SetPos(hg.Vec3(0, 50, 90))
        Main.camera_intro.GetTransform().SetRot(hg.Vec3(0, 0, 0))
        Main.smart_camera.follow_distance = Main.anim_camera_intro_dist.v_start
        Main.smart_camera.lateral_rot = Main.anim_camera_intro_rot.v_start
        Main.smart_camera.setup(SmartCamera.TYPE_FOLLOW, Main.camera_intro, nd)
        Main.display_dark_design = True
        Main.display_Logo = True

    elif Main.intro_anim_id == 0:
        pos, rot, fov = hg.Vec3(13.018, 19.664, 49.265), hg.Vec3(-0.310, 4.065, 0.000), 0.271  # Pilotes sunshine
        Main.camera_intro.GetTransform().SetPos(pos)
        Main.camera_intro.GetTransform().SetRot(rot)
        Main.camera_intro.GetCamera().SetFov(fov)
        Main.smart_camera.setup(SmartCamera.TYPE_FPS, Main.camera_intro)
        Main.scene.SetCurrentCamera(Main.camera_intro)
        Main.display_dark_design = True
        Main.display_logo = False

    elif Main.intro_anim_id == 2:

        Main.display_dark_design = True
        Main.display_logo = False

        keyframes = [

            {"order": 0, "name": "Lateral", "duration": 10, "fade_in": 0.5, "fade_out": 0.5,
             "pos_start": hg.Vec3(-89.067, 17.386, -9.640), "pos_end": hg.Vec3(-88.963, 17.386, -13.672),
             "rot_start": hg.Vec3(0.000, 1.545, 0.000), "rot_end": hg.Vec3(0.000, 1.545, 0.000),
             "fov_start": 0.133, "fov_end": 0.133},

            {"order": 1, "name": "Pilotes sunshine", "duration": 10, "fade_in": 0.5, "fade_out": 0.5,
             "pos_start": hg.Vec3(13.145, 19.664, 49.002), "pos_end": hg.Vec3(12.950, 19.664, 49.531),
             "rot_start": hg.Vec3(-0.300, 3.925, 0.000), "rot_end": hg.Vec3(-0.300, 3.925, 0.000),
             "fov_start": 0.271, "fov_end": 0.271},

            {"order": 2, "name": "Carrier mass", "duration": 10, "fade_in": 0.5, "fade_out": 0.5,
             "pos_start": hg.Vec3(196.515, 34.984, 272.780), "pos_end": hg.Vec3(179.877, 28.862, 262.723),
             "rot_start": hg.Vec3(0.055, 3.755, 0.000), "rot_end": hg.Vec3(0.035, 3.755, 0.000),
             "fov_start": 0.120, "fov_end": 0.120},

            {"order": 3, "name": "missiles", "duration": 10, "fade_in": 0.5, "fade_out": 0.5,
             "pos_end": hg.Vec3(-41.906, 19.598, 40.618), "pos_start": hg.Vec3(-41.881, 19.598, 39.685),
             "rot_start": hg.Vec3(0.010, 1.545, 0.000), "rot_end": hg.Vec3(0.010, 1.545, 0.000),
             "fov_start": 0.039, "fov_end": 0.039},

            {"order": 4, "name": "behind carrier", "duration": 10, "fade_in": 0.5, "fade_out": 0.5,
             "pos_start": hg.Vec3(-7.179, 16.457, 832.241), "pos_end": hg.Vec3(-7.179, 40.457, 832.241),
             "rot_start": hg.Vec3(-0.015, 3.130, 0.000), "rot_end": hg.Vec3(0.015, 3.130, 0.000),
             "fov_start": 0.052, "fov_end": 0.048}
        ]
        strt = int(uniform(0, 1000)) % len(keyframes)
        for kf in keyframes:
            kf["order"] = strt
            strt = (strt + 1) % len(keyframes)
        keyframes.sort(key=lambda p: p["order"])

        Main.smart_camera.setup(SmartCamera.TYPE_CINEMATIC, Main.camera_intro)
        Main.scene.SetCurrentCamera(Main.camera_intro)
        Main.smart_camera.set_keyframes(keyframes)

        # pos,rot,fov = hg.Vec3(-93.775, 17.924, 37.808),hg.Vec3(0.015, 1.540, 0.000),0.145 #Lateral
        # pos,rot,fov = hg.Vec3(13.018,19.664,49.265),hg.Vec3(-0.310,4.065,0.000),0.271 # Pilotes sunshine
        # pos, rot, fov = hg.Vec3(196.515, 34.984, 272.780), hg.Vec3(0.055, 3.755, 0.000), 0.120 # Carrier mass
        # pos, rot, fov = hg.Vec3(-28.444, 19.464, 41.583), hg.Vec3(0.010, 1.545, 0.000), 0.039 # missiles
        # pos, rot, fov = hg.Vec3(-7.179, 16.457, 832.241), hg.Vec3(-0.015, 3.130, 0.000), 0.052 # behind carrier

    Main.scene.SetCurrentCamera(Main.camera_intro)
    Main.t = 0
    Main.fading_cptr = 0
    Main.menu_fading_cptr = 0
    if Main.flag_sfx:
        Main.main_music_state[0].volume = Main.master_sfx_volume
        Main.main_music_state[1].volume = Main.master_sfx_volume
        Main.main_music_source = play_stereo_sound(Main.main_music_ref, Main.main_music_state)

    Main.post_process.setup_fading(3, 1)
    Main.flag_running = True
    return update_menu_phase


def update_menu_phase(dts):
    Main.t += dts
    Main.post_process.update_fading(dts)
    if Main.flag_sfx:
        if Main.post_process.fade_running:
            Main.master_sfx_volume = Main.post_process.fade_f
        set_stereo_volume(Main.main_music_source, Main.master_sfx_volume)

    if Main.intro_anim_id == 1:
        Main.anim_camera_intro_dist.update(Main.t)
        Main.anim_camera_intro_rot.update(Main.t)

        Main.smart_camera.follow_distance = Main.anim_camera_intro_dist.v
        Main.smart_camera.lateral_rot = Main.anim_camera_intro_rot.v

    Main.smart_camera.update(Main.camera_intro, dts)

    for carrier in Main.aircraft_carrier_allies:
        carrier.update_kinetics(dts)

    if Main.display_dark_design:
        Main.spr_design_menu.set_position(0.5 * Main.resolution.x, 0.5 * Main.resolution.y)
        Main.spr_design_menu.set_color(hg.Color(1, 1, 1, 1))
        Main.sprites_display_list.append(Main.spr_design_menu)

    if Main.display_logo:
        Main.spr_logo.set_position(0.5 * Main.resolution.x, 0.5 * Main.resolution.y)
        Main.sprites_display_list.append(Main.spr_logo)

    # fade in:
    fade_in_delay = 2.
    Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)

    if Main.fading_cptr >= fade_in_delay:
        # Start infos:
        tps = hg.time_to_sec_f(hg.GetClock())
        menu_fade_in_delay = 1
        Main.menu_fading_cptr = min(menu_fade_in_delay, Main.menu_fading_cptr + dts)

        f = Main.menu_fading_cptr / menu_fade_in_delay

        yof7 = -0.15

        Overlays.add_text2D("FLIGHT SIMULATOR V0.8", hg.Vec2(0.5, 800 / 900 - 0.08), 0.035, hg.Color.White * f, Main.title_font, hg.DTHA_Center)
        Overlays.add_text2D("by sserver", hg.Vec2(0.5, 770 / 900 - 0.08), 0.025, hg.Color.White * f, Main.hud_font, hg.DTHA_Center)

        Missions.display_mission_title(Main, f, dts, yof7)

        Overlays.add_text2D("Hit space or Start", hg.Vec2(0.5, 611 / 900 + yof7), 0.025, hg.Color(1, 1, 1, (0.7 + sin(tps * 5) * 0.3)) * f, Main.title_font, hg.DTHA_Center)

        s = 0.015
        x = 470 / 1600
        y = 520 + yof7 * 900
        c = hg.Color(1., 0.9, 0.3, 1) * f
        if Main.flag_vr:
            Overlays.add_text2D("Recenter view", hg.Vec2(x, (y+20) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Throttle", hg.Vec2(x, (y) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Pitch", hg.Vec2(x, (y - 20) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Roll", hg.Vec2(x, (y - 40) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Yaw", hg.Vec2(x, (y - 60) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Gun", hg.Vec2(x, (y - 80) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Missiles", hg.Vec2(x, (y - 100) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Target selection", hg.Vec2(x, (y - 120) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Speedbrakes", hg.Vec2(x, (y - 140) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Flaps", hg.Vec2(x, (y - 160) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Afterburner", hg.Vec2(x, (y - 180) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Gear Up/Down", hg.Vec2(x, (y - 200) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Reset game", hg.Vec2(x, (y - 220) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Set View", hg.Vec2(x, (y - 240) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Zoom", hg.Vec2(x, (y - 260) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Plane select (multi ally missions)", hg.Vec2(x, (y - 280) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Activate IA", hg.Vec2(x, (y - 300) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Activate User control", hg.Vec2(x, (y - 320) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("HUD ON / OFF", hg.Vec2(x, (y - 340) / 900), s, c, Main.hud_font)


        c2 = hg.Color.Grey * f

        # Keyboard:
        x = 815 / 1600
        c = hg.Color.White * f
        Overlays.add_text2D("KBD commands", hg.Vec2(x, (y+40) / 900), s, c2, Main.hud_font)
        if Main.flag_vr:
            Overlays.add_text2D("F11", hg.Vec2(x, (y+20) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Home / End", hg.Vec2(x, (y) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Up / Down", hg.Vec2(x, (y - 20) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Right / Left", hg.Vec2(x, (y - 40) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Suppr / Page down", hg.Vec2(x, (y - 60) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("ENTER", hg.Vec2(x, (y - 80) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("F1", hg.Vec2(x, (y - 100) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("T", hg.Vec2(x, (y - 120) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("B", hg.Vec2(x, (y - 140) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("C / V", hg.Vec2(x, (y - 160) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Auto", hg.Vec2(x, (y - 180) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("G", hg.Vec2(x, (y - 200) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Tab", hg.Vec2(x, (y - 220) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("2/3/4/8/6/5/7/9", hg.Vec2(x, (y - 240) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Insert / Page Up", hg.Vec2(x, (y - 260) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("Numeric pad : 1", hg.Vec2(x, (y - 280) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("I", hg.Vec2(x, (y - 300) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("U", hg.Vec2(x, (y - 320) / 900), s, c, Main.hud_font)
        Overlays.add_text2D("F10", hg.Vec2(x, (y - 340) / 900), s, c, Main.hud_font)

        # Paddle
        if Main.flag_paddle:
            x = 990 / 1600
            Overlays.add_text2D("Gamepad commands", hg.Vec2(x, (y+40) / 900), s, c2, Main.hud_font)
            Overlays.add_text2D("Left stick up",  hg.Vec2(x, (y) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Right stick vertical", hg.Vec2(x, (y - 20) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Right stick horizontal", hg.Vec2(x, (y - 40) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Left stick horizontal", hg.Vec2(x, (y - 60) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("A", hg.Vec2(x, (y - 80) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("X", hg.Vec2(x, (y - 100) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("N/A", hg.Vec2(x, (y - 120) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Left stick down", hg.Vec2(x, (y - 140) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("D-Pad Right / Left", hg.Vec2(x, (y - 160) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Auto",  hg.Vec2(x, (y - 180) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("Y", hg.Vec2(x, (y - 200) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("B", hg.Vec2(x, (y - 300) / 900), s, c, Main.hud_font)
            Overlays.add_text2D("N/A", hg.Vec2(x, (y - 320) / 900), s, c, Main.hud_font)

    if not Main.fading_to_next_state:
        f_start = False
        if Main.keyboard.Pressed(hg.K_Space):
            Main.control_mode = AircraftUserControlDevice.CM_KEYBOARD
            f_start = True
        elif Main.flag_paddle:
            if Main.gamepad.Pressed(hg.GB_Start):
                Main.control_mode = AircraftUserControlDevice.CM_GAMEPAD
                f_start = True
        elif Main.flag_generic_controller:
            if Main.generic_controller.Down(1):
                Main.control_mode = AircraftUserControlDevice.CM_LOGITECH_ATTACK_3
                f_start = True                
        if f_start:
            Main.post_process.setup_fading(1, -1)
            Main.fading_to_next_state = True
    else:
        if not Main.post_process.fade_running:
            Main.destroy_players()
            init_main_phase()
            return update_main_phase
    return update_menu_phase


# =================================== IN GAME =============================================

def init_main_phase():
    Main.flag_running = False
    Main.fading_to_next_state = False
    Main.post_process.setup_fading(1, 1)
    #hg.StopAllSources()
    Main.destroy_sfx()
    ParticlesEngine.reset_engines()
    Destroyable_Machine.reset_machines()
    mission = Missions.get_current_mission()

    mission.setup_players(Main)

    n_aircrafts = Main.num_players_allies + Main.num_players_ennemies
    n_missile_launchers = Main.num_missile_launchers_allies + Main.num_missile_launchers_ennemies
    n_missiles = 0
    for aircraft in Main.players_allies:
        n_missiles += aircraft.get_num_missiles_slots()
    for aircraft in Main.players_ennemies:
        n_missiles += aircraft.get_num_missiles_slots()

    HUD_Radar.setup_plots(Main.resolution, n_aircrafts, n_missiles, mission.allies_carriers + mission.ennemies_carriers, n_missile_launchers)

    #Main.setup_weaponery()

    Main.num_start_frames = 10
    Main.timestamp = 0
    Main.flag_running = True
    return update_main_phase


def update_main_phase(dts):

    Main.timestamp += 1
    if not Main.flag_renderless:
        Main.post_process.update_fading(dts)
        if Main.flag_sfx:
            if Main.post_process.fade_running:
                Main.master_sfx_volume = Main.post_process.fade_f

        if Main.flag_control_views:
            Main.control_views(Main.keyboard)

        if Main.flag_display_HUD:
            if Main.user_aircraft is not None:
                if Main.user_aircraft.type == Destroyable_Machine.TYPE_AIRCRAFT:
                    HUD_Aircraft.update(Main, Main.user_aircraft, Main.destroyables_list)
                elif Main.user_aircraft.type == Destroyable_Machine.TYPE_MISSILE_LAUNCHER:
                    HUD_MissileLauncher.update(Main, Main.user_aircraft, Main.destroyables_list)

            if Main.flag_display_selected_aircraft and Main.selected_aircraft is not None:
                HUD_MissileTarget.display_selected_target(Main, Main.selected_aircraft)

        if Main.flag_display_landing_trajectories:
            if Main.user_aircraft is not None:
                ia_ctrl = Main.user_aircraft.get_device("IAControlDevice")
                if ia_ctrl is not None:
                    Main.display_landing_trajectory(ia_ctrl.IA_landing_target)
                    Main.display_landing_projection(Main.user_aircraft)
        if Main.flag_display_machines_bounding_boxes:
            for machine in Destroyable_Machine.machines_list:
                if machine.is_activated:
                    if machine.bounding_boxe is not None:
                        matrix = machine.get_parent_node().GetTransform().GetWorld()
                        Overlays.add_line(machine.bound_front * matrix, (machine.bound_front + hg.Vec3(0, 0, 1)) * matrix, hg.Color.Blue, hg.Color.Blue)
                        Overlays.add_line(machine.bound_back * matrix, (machine.bound_back + hg.Vec3(0, 0, -1)) * matrix, hg.Color.Blue, hg.Color.Blue)
                        Overlays.add_line(machine.bound_up * matrix, (machine.bound_up + hg.Vec3(0, 1, 0)) * matrix, hg.Color.Green, hg.Color.Green)
                        Overlays.add_line(machine.bound_down * matrix, (machine.bound_down + hg.Vec3(0, -1, 0)) * matrix, hg.Color.Green, hg.Color.Green)
                        Overlays.add_line(machine.bound_right * matrix, (machine.bound_right + hg.Vec3(1, 0, 0)) * matrix, hg.Color.Red, hg.Color.Red)
                        Overlays.add_line(machine.bound_left * matrix, (machine.bound_left + hg.Vec3(-1, 0, 0)) * matrix, hg.Color.Red, hg.Color.Red)
                        Overlays.display_boxe(machine.get_world_bounding_boxe(), hg.Color.Yellow)
    else:
        if Main.user_aircraft is not None and Main.flag_display_radar_in_renderless:
            HUD_Radar.update(Main, Main.user_aircraft, Main.destroyables_list)

    # Destroyable_Machines physics & movements update
    Main.update_kinetics(dts)

    # Update sfx
    if Main.flag_sfx:
        for sfx in Main.players_sfx: sfx.update_sfx(Main, dts)
        for sfx in Main.missiles_sfx: sfx.update_sfx(Main, dts)

    camera_noise_level = 0

    if Main.user_aircraft is not None:

        if Main.user_aircraft.type == Destroyable_Machine.TYPE_AIRCRAFT:
            acc = Main.user_aircraft.get_linear_acceleration()
            camera_noise_level = max(0, Main.user_aircraft.get_linear_speed() * 3.6 / 2500 * 0.1 + pow(min(1, abs(acc / 7)), 2) * 1)
            if Main.user_aircraft.post_combustion:
                camera_noise_level += 0.1

        if Main.player_view_mode == SmartCamera.TYPE_FIX:
            cam = Main.camera_cokpit
        else:
            cam = Main.camera

        if Main.keyboard.Pressed(hg.K_Y):
            flag = Main.user_aircraft.get_custom_physics_mode()
            Main.user_aircraft.set_custom_physics_mode(not flag)

    else:
        cam = Main.camera_fps

    if Main.satellite_view:
        cam = Main.satellite_camera

    Main.smart_camera.update(cam, dts, camera_noise_level)

    mission = Missions.get_current_mission()

    if Main.keyboard.Pressed(hg.K_L):
        Destroyable_Machine.flag_update_particles = not Destroyable_Machine.flag_update_particles

    if Main.keyboard.Pressed(hg.K_R):
        Main.set_renderless_mode(not Main.flag_renderless)
    if get_connected()[0]:
        if get_button_values(get_state(0))['BACK'] and not get_button_values(get_state(0))['START']:
            Main.set_renderless_mode(False)
            mission.aborted = True
            init_end_phase()
            return update_end_phase
    if Main.keyboard.Pressed(hg.K_Tab):
        Main.set_renderless_mode(False)
        mission.aborted = True
        init_end_phase()
        return update_end_phase
    elif mission.end_test(Main):
        init_end_phase()
        return update_end_phase

    return update_main_phase


# =================================== END GAME =============================================

def init_end_phase():
    Main.set_renderless_mode(False)
    Main.flag_running = False
    Main.deactivate_cockpit_view()
    Main.satellite_view = False
    Main.end_state_timer = 20

    if Main.user_aircraft is not None:

        uctrl = Main.user_aircraft.get_device("UserControlDevice")
        if uctrl is not None:
            uctrl.deactivate()

        ia = Main.user_aircraft.get_device("IAControlDevice")
        if ia is not None:
            ia.activate()

        Main.user_aircraft = None

    mission = Missions.get_current_mission()

    aircraft = None
    if mission.failed:
        for player in Main.players_ennemies:
            if not player.wreck:
                aircraft = player
                break
        if aircraft is None:
            aircraft = Main.players_allies[0]
    else:
        for player in Main.players_allies:
            if not player.wreck:
                aircraft = player
                break
        if aircraft is None:
            aircraft = Main.players_allies[0]

    Main.end_phase_following_aircraft = aircraft
    Main.smart_camera.setup(SmartCamera.TYPE_FOLLOW, Main.camera, aircraft.get_parent_node())
    Main.scene.SetCurrentCamera(Main.camera)
    Main.flag_running = True
    return update_end_phase


def update_end_phase(dts):

    Main.smart_camera.update(Main.camera, dts)

    Main.update_kinetics(dts)

    if Main.flag_sfx:
        for sfx in Main.players_sfx: sfx.update_sfx(Main, dts)
        for sfx in Main.missiles_sfx: sfx.update_sfx(Main, dts)

    if Main.flag_display_selected_aircraft and Main.selected_aircraft is not None:
        HUD_MissileTarget.display_selected_target(Main, Main.selected_aircraft)

    mission = Missions.get_current_mission()
    mission.update_end_phase(Main, dts)

    if not Main.fading_to_next_state:
        if Main.end_phase_following_aircraft.flag_destroyed or Main.end_phase_following_aircraft.wreck:
            Main.end_state_timer -= dts
        if get_connected()[0]:
            if get_button_values(get_state(0))['BACK'] and not get_button_values(get_state(0))['START']:
                Main.post_process.setup_fading(1, -1)
                Main.fading_to_next_state = True
        if Main.keyboard.Pressed(hg.K_Tab) or Main.end_state_timer < 0 or Main.end_phase_following_aircraft.flag_landed:
            Main.post_process.setup_fading(1, -1)
            Main.fading_to_next_state = True
    else:
        Main.post_process.update_fading(dts)
        if Main.flag_sfx:
            Main.master_sfx_volume = Main.post_process.fade_f
        if not Main.post_process.fade_running:
            mission.reset()
            Main.destroy_players()
            init_menu_phase()
            return update_menu_phase

    return update_end_phase
# --------------- Inline arguments handler
def get_resource_path(relative_path):
        try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS
        except Exception:
                base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
def run_command(exe):
        def execute_com(command):
                p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                return iter(p.stdout.readline, b'')
        for line in execute_com(exe):
                txt = str(line)
                if "Progress" in txt:
                        try:
                                percent = int(re.findall('\d*%', txt)[0].split("%")[0])
                        except:
                                break
                        prog['value']=float(percent)
                        style.configure('text.Horizontal.TProgressbar',
                        text='{:g} %'.format(percent))
                        root.update()
for i in range(len(sys.argv)):
    cmd = sys.argv[i]
    if cmd == "network_port":
        try:
            nwport = int(sys.argv[i+1])
            dogfight_network_port = nwport
            print("Network port:" + str(nwport))
        except:
            print("ERROR !!! Bad port format - network port must be a valid number !!!")
        i += 1

# ---------------- Read config file:

file_name="./config.json"

file = open(file_name, "r")
json_script = file.read()
file.close()
if json_script != "":
    script_parameters = json.loads(json_script)
    Main.flag_OpenGL = script_parameters["OpenGL"]
    Main.flag_vr = script_parameters["VR"]
    Main.flag_fullscreen = script_parameters["FullScreen"]
    Main.resolution.x = script_parameters["Resolution"][0]
    Main.resolution.y = script_parameters["Resolution"][1]
    Main.antialiasing = script_parameters["AntiAliasing"]
    Main.flag_shadowmap = script_parameters["ShadowMap"]

# If the VR is enabled the main window becomes useless
# so we downsize it.
if Main.flag_vr:
    Main.resolution.x = 256
    Main.resolution.y = 256

# --------------- VR mode only under DirectX
if Main.flag_OpenGL:
    if Main.flag_vr:
        print("WARNING - VR mode only available under DirectX (OpenGL : False in Config.json) - VR is turned to OFF")
        Main.flag_vr = False

# --------------- Compile assets:
root=Tk()
root.resizable(False, False)
root.iconbitmap(get_resource_path('plane.ico'))
root.protocol('WM_DELETE_WINDOW', sys.exit)
root.geometry('300x100+'+str(int(root.winfo_screenwidth()))+'+'+str(int(root.winfo_screenheight())))
root.title('Compiling...')
root.attributes('-topmost', True)
style = Style(root)
style.layout('text.Horizontal.TProgressbar',
             [('Horizontal.Progressbar.trough',
               {'children': [('Horizontal.Progressbar.pbar',
                              {'side': 'left', 'sticky': 'ns'})],
                'sticky': 'nswe'}),
              ('Horizontal.Progressbar.label', {'sticky': ''})])
              # ,lightcolor=None,bordercolo=None,darkcolor=None
style.configure('text.Horizontal.TProgressbar', text='0 %')
Label(root, text='Compiling assets...').pack()
prog=Progressbar(root, maximum=100, mode='determinate', length=200, style='text.Horizontal.TProgressbar')
prog.pack()
root.update()
if sys.platform == "linux" or sys.platform == "linux2":
    assetc_cmd = [path.join(getcwd(), "../", "bin", "assetc", "assetc"), "assets", "-quiet", "-progress"]
    run_command(assetc_cmd)
else:
    if Main.flag_OpenGL:
        run_command("./assetc/assetc assets -api GL -quiet -progress")
    else:
        run_command("./assetc/assetc assets -quiet -progress")
root.destroy()
# --------------- Init system

hg.InputInit()
hg.WindowSystemInit()

hg.SetLogDetailed(False)

res_x, res_y = int(Main.resolution.x), int(Main.resolution.y)

hg.AddAssetsFolder(Main.assets_compiled)

# ------------------- Setup output window
def get_monitor_mode(width, height):
    monitors = hg.GetMonitors()
    for i in range(monitors.size()):
        monitor = monitors.at(i)
        f, monitorModes = hg.GetMonitorModes(monitor)
        if f:
            for j in range(monitorModes.size()):
                mode = monitorModes.at(j)
                if mode.rect.ex == width and mode.rect.ey == height:
                    print("get_monitor_mode() : Width %d Height %d" % (mode.rect.ex, mode.rect.ey))
                    return monitor, j
    return None, 0


Main.win = None
if Main.flag_fullscreen:
    monitor, mode_id = get_monitor_mode(res_x, res_y)
    if monitor is not None:
        Main.win = hg.NewFullscreenWindow(monitor, mode_id)

if Main.win is None:
    Main.win = hg.NewWindow(res_x, res_y)

if Main.flag_OpenGL:
    hg.RenderInit(Main.win, hg.RT_OpenGL)
else:
    hg.RenderInit(Main.win)

alias_modes = [hg.RF_MSAA2X, hg.RF_MSAA4X, hg.RF_MSAA8X, hg.RF_MSAA16X]
aa = alias_modes[min(3, floor(log(Main.antialiasing) / log(2)) - 1)]
hg.RenderReset(res_x, res_y, aa | hg.RF_MaxAnisotropy)

# -------------------- OpenVR initialization

if Main.flag_vr:
    if not Main.setup_vr():
        sys.exit()

# ------------------- Imgui for UI

imgui_prg = hg.LoadProgramFromAssets('core/shader/imgui')
imgui_img_prg = hg.LoadProgramFromAssets('core/shader/imgui_image')
hg.ImGuiInit(10, imgui_prg, imgui_img_prg)


# --------------------- Setup dogfight sim
hg.AudioInit()
Main.init_game()

node = Main.scene.GetNode("platform.S400")
nm = node.GetName()

# rendering pipeline
Main.pipeline = hg.CreateForwardPipeline()
hg.ResetClock()

# ------------------- Setup state:
Main.current_state = init_menu_phase()

# ------------------- Main loop:
while not Main.flag_exit:
        Main.update_inputs()
        if (not Main.flag_client_update_mode) or ((not Main.flag_renderless) and Main.flag_client_ask_update_scene):
                Main.update()
        else:
                time.sleep(1 / 120)
        Main.update_window()
if Main.flag_network_mode:
    stop_server()

hg.StopAllSources()
hg.AudioShutdown()

hg.RenderShutdown()
hg.DestroyWindow(Main.win)
