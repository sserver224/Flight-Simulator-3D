import harfang as hg
import platform
import json
from random import *
from math import *
from XInput import *
class WaterReflection():
        def __init__(self, plus, scene, resolution: hg.Vector2, texture_width=256):

                renderer = plus.GetRenderer()
                render_system = plus.GetRenderSystem()

                # Parameters:
                self.color = hg.Color(1, 1, 0, 1)
                self.reflect_level = 0.75

                # Shaders:
                # self.shader_water_reflection = render_system.LoadSurfaceShader("assets/shaders/water_reflection.isl")

                # Reflection plane, just to get normal & origine:
                # self.plane=plus.AddPlane(scene, hg.Matrix4.TransformationMatrix(hg.Vector3(0,0,0), hg.Vector3(radians(0), radians(0), radians(0))), 1, 1)
                # self.plane.SetName("Water_Reflection_Plane")

                # Création des textures de rendu:
                tex_res = hg.Vector2(texture_width, texture_width * resolution.y / resolution.x)
                self.render_texture = renderer.NewTexture()
                renderer.CreateTexture(self.render_texture, int(tex_res.x), int(tex_res.y), hg.TextureRGBA8, hg.TextureNoAA, 0,
                                                           False)
                self.render_depth_texture = renderer.NewTexture()
                renderer.CreateTexture(self.render_depth_texture, int(tex_res.x), int(tex_res.y), hg.TextureDepth,
                                                           hg.TextureNoAA, 0, False)

                # Création des frameBuffer objects:
                self.render_target = renderer.NewRenderTarget()
                renderer.CreateRenderTarget(self.render_target)
                renderer.SetRenderTargetColorTexture(self.render_target, self.render_texture)
                renderer.SetRenderTargetDepthTexture(self.render_target, self.render_depth_texture)

                self.projection_matrix_mem = None
                self.view_matrix_mem = None
                self.projection_matrix_ortho = None
                # Reflection camera:
                self.camera_reflect = plus.AddCamera(scene, hg.Matrix4.TranslationMatrix(hg.Vector3(0, -2, 0)))
                self.camera_reflect.SetName("Camera.water_reflection")

                self.clear_reflect_map(plus)

        @staticmethod
        def get_plane_projection_factor(p: hg.Vector3, plane_origine: hg.Vector3, plane_normal: hg.Vector3):
                d = -plane_normal.x * plane_origine.x - plane_normal.y * plane_origine.y - plane_normal.z * plane_origine.z
                return -plane_normal.x * p.x - plane_normal.y * p.y - plane_normal.z * p.z - d

        def clear_reflect_map(self, plus):
                renderer = plus.GetRenderer()
                renderer.SetRenderTarget(self.render_target)
                renderer.Clear(hg.Color(0., 0., 0., 0.))

        def render(self, plus, scene, camera, disable_render_scripts=False, mat_camera=None):
                renderer = plus.GetRenderer()
                # Clipping plane:
                # mat=self.plane.GetTransform().GetWorld()
                # plane_pos=mat.GetTranslation()
                # plane_normal=mat.GetY()
                # renderer.SetClippingPlane(plane_pos+plane_normal*0.01, plane_normal)

                plane_pos = hg.Vector3(0, 0, 0)
                plane_normal = hg.Vector3(0, 1, 0)

                # Camera reflect:
                if mat_camera is not None:
                        mat = mat_camera
                else:
                        mat = camera.GetTransform().GetWorld()
                pos = mat.GetTranslation()
                t = self.get_plane_projection_factor(pos, plane_pos, plane_normal)
                pos_reflect = pos + plane_normal * 2 * t
                xAxis = mat.GetX()
                zAxis = mat.GetZ()
                px = pos + xAxis
                tx = self.get_plane_projection_factor(px, plane_pos, plane_normal)
                x_reflect = px + plane_normal * 2 * tx - pos_reflect
                z_reflect = hg.Reflect(zAxis, plane_normal)
                y_reflect = hg.Cross(z_reflect, x_reflect)
                mat.SetTranslation(pos_reflect)
                mat.SetX(x_reflect)
                mat.SetY(y_reflect)
                mat.SetZ(z_reflect)
                self.camera_reflect.GetTransform().SetWorld(mat)
                scene.SetCurrentCamera(self.camera_reflect)
                cam_org = camera.GetCamera()
                cam = self.camera_reflect.GetCamera()
                cam.SetZoomFactor(cam_org.GetZoomFactor())
                cam.SetZNear(cam_org.GetZNear())
                cam.SetZFar(cam_org.GetZFar())
                # Render target:
                vp_mem = renderer.GetViewport()
                rect = self.render_texture.GetRect()
                renderer.SetViewport(rect)
                renderer.SetRenderTarget(self.render_target)
                renderer.Clear(hg.Color(0., 0., 0., 0.))

                if disable_render_scripts:
                        enabled_list = []
                        render_scripts = scene.GetComponents("RenderScript")
                        for rs in render_scripts:
                                enabled_list.append(rs.GetEnabled())
                                rs.SetEnabled(False)

                scene.Commit()
                scene.WaitCommit()
                plus.UpdateScene(scene)

                # Update reflection texture:
                # material = self.plane.GetObject().GetGeometry().GetMaterial(0)
                # material.SetTexture("reflect_map", self.render_texture)
                # material.SetFloat("reflect_level", self.reflect_level)
                # material.SetFloat4("color", self.color.r,self.color.g,self.color.b,self.color.a)

                # System restauration
                scene.SetCurrentCamera(camera)
                renderer.ClearRenderTarget()
                # renderer.ClearClippingPlane()
                renderer.SetViewport(vp_mem)

                if disable_render_scripts:
                        for rs in enumerate(render_scripts):
                                if enabled_list[rs[0]]:
                                        rs[1].SetEnabled(True)

                scene.Commit()
                scene.WaitCommit()

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
class Particle:
        def __init__(self, node: hg.Node):
                self.node = node
                self.age = -1
                self.v_move = hg.Vector3(0, 0, 0)
                self.delay = 0
                self.scale = 1
                self.rot_speed = hg.Vector3(0, 0, 0)

        def kill(self):
                self.age=-1
                self.node.SetEnabled(False)
class SeaRender:
        def __init__(self, plus, scene, render_script: hg.RenderScript = None):
                self.render_script = render_script
                self.sun_light = scene.GetNode("Sun")
                self.sky_light = scene.GetNode("SkyLigth")

                self.sea_scale = hg.Vector3(0.02, 3, 0.005)

                self.zenith_color = hg.Color(17. / 255., 56. / 255., 155. / 255., 1.)
                self.horizon_N_color = hg.Color(76. / 255., 128. / 255., 255 / 255., 1.)
                self.horizon_S_color = hg.Color(76. / 255., 128. / 255., 255 / 255., 1.)
                self.sea_color = hg.Color(19 / 255., 39. / 255., 89. / 255., 1.)
                self.horizon_line_color = hg.Color(1, 1, 1, 1)
                self.horizon_line_size = 40
                self.sea_reflection = 0.5
                renderer = plus.GetRenderer()

                self.sea_filtering = 0  # 1 to activate sea texture filtering
                self.render_scene_reflection = False

                self.max_filter_samples = 3
                self.filter_precision = 10

                self.shader = renderer.LoadShader("assets/shaders/sky_sea_render_optim.isl")

                self.noise_texture_1 = renderer.LoadTexture("assets/textures/noise.png")
                self.noise_texture_2 = renderer.LoadTexture("assets/textures/noise_3.png")
                self.noise_displacement_texture = renderer.LoadTexture("assets/textures/noise_2.png")
                self.stream_texture = renderer.LoadTexture("assets/textures/stream.png")
                self.clouds_map = renderer.LoadTexture("assets/textures/clouds_map.png")

                self.clouds_scale = hg.Vector3(1000., 0.1, 1000.)
                self.clouds_altitude = 1000.
                self.clouds_absorption = 0.1

                self.tex_sky_N = renderer.LoadTexture("assets/skymaps/clouds.png")
                self.tex_sky_N_intensity = 1
                self.zenith_falloff = 4

                self.reflect_map = None
                self.reflect_map_depth = None
                self.reflect_offset = 50

                self.render_sea = True

        def load_json_script(self, file_name="assets/scripts/sea_parameters.json"):
                json_script = hg.GetFilesystem().FileToString(file_name)
                if json_script != "":
                        script_parameters = json.loads(json_script)
                        self.horizon_N_color = list_to_color(script_parameters["horizon_N_color"])
                        self.horizon_S_color = list_to_color(script_parameters["horizon_S_color"])
                        self.zenith_color = list_to_color(script_parameters["zenith_color"])
                        self.zenith_falloff = script_parameters["zenith_falloff"]
                        self.tex_sky_N_intensity = script_parameters["tex_sky_N_intensity"]
                        self.horizon_line_color = list_to_color(script_parameters["horizon_line_color"])
                        self.sea_color = list_to_color(script_parameters["sea_color"])
                        self.sea_scale = list_to_vec3(script_parameters["sea_scale"])
                        self.sea_reflection = script_parameters["sea_reflection"]
                        self.horizon_line_size = script_parameters["horizon_line_size"]
                        self.sea_filtering = script_parameters["sea_filtering"]
                        self.max_filter_samples = script_parameters["max_filter_samples"]
                        self.filter_precision = script_parameters["filter_precision"]
                        self.clouds_scale = list_to_vec3(script_parameters["clouds_scale"])
                        self.clouds_altitude = script_parameters["clouds_altitude"]
                        self.clouds_absorption = script_parameters["clouds_absorption"]
                        self.reflect_offset = script_parameters["reflect_offset"]
                        self.render_scene_reflection = script_parameters["render_scene_reflection"]

        def save_json_script(self, output_filename="assets/scripts/sea_parameters.json"):
                script_parameters = {"horizon_N_color": color_to_list(self.horizon_N_color),
                        "horizon_S_color": color_to_list(self.horizon_S_color),
                        "horizon_line_color": color_to_list(self.horizon_line_color),
                        "zenith_color": color_to_list(self.zenith_color), "zenith_falloff": self.zenith_falloff,
                        "tex_sky_N_intensity": self.tex_sky_N_intensity, "sea_color": color_to_list(self.sea_color),
                        "sea_reflection": self.sea_reflection, "horizon_line_size": self.horizon_line_size,
                        "sea_scale": vec3_to_list(self.sea_scale), "sea_filtering": self.sea_filtering,
                        "max_filter_samples": self.max_filter_samples, "filter_precision": self.filter_precision,
                        "clouds_scale": vec3_to_list(self.clouds_scale), "clouds_altitude": self.clouds_altitude,
                        "clouds_absorption": self.clouds_absorption, "reflect_offset": self.reflect_offset,
                        "render_scene_reflection": self.render_scene_reflection}
                json_script = json.dumps(script_parameters, indent=4)
                return hg.GetFilesystem().StringToFile(output_filename, json_script)

        def enable_render_sea(self, value):
                self.render_sea = value
                if value:
                        self.render_script.Set("render_sea", 1)
                else:
                        self.render_script.Set("render_sea", 0)

        def update_render_script(self, scene, resolution: hg.Vector2, time):
                if self.render_sea:
                        self.render_script.Set("render_sea", 1)
                else:
                        self.render_script.Set("render_sea", 0)
                self.render_script.Set("zenith_color", self.zenith_color)
                self.render_script.Set("horizon_N_color", self.horizon_N_color)
                self.render_script.Set("horizon_S_color", self.horizon_S_color)
                self.render_script.Set("zenith_falloff", self.zenith_falloff)
                self.render_script.Set("horizon_line_color", self.horizon_line_color)
                self.render_script.Set("sea_color", self.sea_color)
                l_color = self.sun_light.GetLight().GetDiffuseColor()
                self.render_script.Set("sun_color", l_color)
                amb = hg.Color(scene.GetEnvironment().GetAmbientColor()) * scene.GetEnvironment().GetAmbientIntensity()
                self.render_script.Set("ambient_color", amb)
                self.render_script.Set("horizon_line_size", self.horizon_line_size)
                self.render_script.Set("tex_sky_N_intensity", self.tex_sky_N_intensity)
                self.render_script.Set("resolution", resolution)
                self.render_script.Set("clouds_scale", self.clouds_scale)
                self.render_script.Set("clouds_altitude", self.clouds_altitude)
                self.render_script.Set("clouds_absorption", self.clouds_absorption)

                camera = scene.GetCurrentCamera()
                pos = camera.GetTransform().GetPreviousWorld().GetTranslation()
                camera = camera.GetCamera()
                self.render_script.Set("sea_reflection", self.sea_reflection)
                self.render_script.Set("sea_filtering", self.sea_filtering)
                self.render_script.Set("max_filter_samples", self.max_filter_samples)
                self.render_script.Set("filter_precision", self.filter_precision)
                l_dir = self.sun_light.GetTransform().GetWorld().GetRotationMatrix().GetZ()
                self.render_script.Set("sun_dir", l_dir)

                self.render_script.Set("sea_scale", self.sea_scale)
                self.render_script.Set("time_clock", time)
                self.render_script.Set("cam_pos", pos)
                self.render_script.Set("z_near", camera.GetZNear())
                self.render_script.Set("z_far", camera.GetZFar())
                self.render_script.Set("zoom_factor", camera.GetZoomFactor())

                self.render_script.Set("reflect_map", self.reflect_map)
                self.render_script.Set("reflect_map_depth", self.reflect_map_depth)
                self.render_script.Set("reflect_offset", self.reflect_offset)

                self.render_script.Set("tex_sky_N", self.tex_sky_N)
                self.render_script.Set("noise_texture_1", self.noise_texture_1)
                self.render_script.Set("noise_texture_2", self.noise_texture_2)
                self.render_script.Set("noise_displacement_texture", self.noise_displacement_texture)
                self.render_script.Set("stream_texture", self.stream_texture)
                self.render_script.Set("clouds_map", self.clouds_map)
                if self.render_scene_reflection:
                        self.render_script.Set("scene_reflect", 1)
                else:
                        self.render_script.Set("scene_reflect", 0)

        def update_shader(self, plus, scene, resolution, time):
                camera = scene.GetCurrentCamera()
                renderer = plus.GetRenderer()
                renderer.EnableDepthTest(True)
                renderer.EnableDepthWrite(True)
                renderer.EnableBlending(False)

                renderer.SetShader(self.shader)
                renderer.SetShaderInt("sea_filtering", self.sea_filtering)
                renderer.SetShaderInt("max_samples", self.max_filter_samples)
                renderer.SetShaderFloat("filter_precision", self.filter_precision)
                renderer.SetShaderFloat("reflect_offset", self.reflect_offset)
                renderer.SetShaderTexture("tex_sky_N", self.tex_sky_N)
                renderer.SetShaderTexture("reflect_map", self.reflect_map)
                renderer.SetShaderTexture("reflect_map_depth", self.reflect_map_depth)
                renderer.SetShaderTexture("noise_texture_1", self.noise_texture_1)
                renderer.SetShaderTexture("noise_texture_2", self.noise_texture_2)
                renderer.SetShaderTexture("displacement_texture", self.noise_displacement_texture)
                renderer.SetShaderTexture("stream_texture", self.stream_texture)
                renderer.SetShaderTexture("clouds_map", self.clouds_map)
                renderer.SetShaderFloat3("clouds_scale", 1. / self.clouds_scale.x, self.clouds_scale.y,
                                                                 1. / self.clouds_scale.z)
                renderer.SetShaderFloat("clouds_altitude", self.clouds_altitude)
                renderer.SetShaderFloat("clouds_absorption", self.clouds_absorption)
                # renderer.SetShaderFloat2("stream_scale",self.stream_scale.x,.y)

                renderer.SetShaderFloat2("resolution", resolution.x, resolution.y)
                renderer.SetShaderFloat("focal_distance", camera.GetCamera().GetZoomFactor())
                renderer.SetShaderFloat("tex_sky_N_intensity", self.tex_sky_N_intensity)
                renderer.SetShaderFloat("zenith_falloff", self.zenith_falloff)
                cam = camera.GetTransform().GetPreviousWorld()
                pos = cam.GetTranslation()
                renderer.SetShaderFloat3("cam_position", pos.x, pos.y, pos.z)
                renderer.SetShaderMatrix3("cam_normal", cam.GetRotationMatrix())
                renderer.SetShaderFloat3("sea_scale", 1 / self.sea_scale.x, self.sea_scale.y, 1 / self.sea_scale.z)
                renderer.SetShaderFloat("time", time)
                renderer.SetShaderFloat3("zenith_color", self.zenith_color.r, self.zenith_color.g, self.zenith_color.b)
                renderer.SetShaderFloat3("horizonH_color", self.horizon_N_color.r, self.horizon_N_color.g,
                                                                 self.horizon_N_color.b)
                renderer.SetShaderFloat3("horizonL_color", self.horizon_S_color.r, self.horizon_S_color.g,
                                                                 self.horizon_S_color.b)
                renderer.SetShaderFloat3("sea_color", self.sea_color.r, self.sea_color.g, self.sea_color.b)
                renderer.SetShaderFloat3("horizon_line_color", self.horizon_line_color.r, self.horizon_line_color.g,
                                                                 self.horizon_line_color.b)
                renderer.SetShaderFloat("sea_reflection", self.sea_reflection)
                renderer.SetShaderFloat("horizon_line_size", self.horizon_line_size)

                l_dir = self.sun_light.GetTransform().GetWorld().GetRotationMatrix().GetZ()

                renderer.SetShaderFloat3("sun_dir", l_dir.x, l_dir.y, l_dir.z)
                l_couleur = self.sun_light.GetLight().GetDiffuseColor()
                renderer.SetShaderFloat3("sun_color", l_couleur.r, l_couleur.g, l_couleur.b)
                amb = hg.Color(scene.GetEnvironment().GetAmbientColor()) * scene.GetEnvironment().GetAmbientIntensity()
                renderer.SetShaderFloat3("ambient_color", amb.r, amb.g, amb.b)
                renderer.SetShaderFloat2("zFrustum", camera.GetCamera().GetZNear(), camera.GetCamera().GetZFar())
                if self.render_scene_reflection:
                        renderer.SetShaderInt("scene_reflect", 1)
                else:
                        renderer.SetShaderInt("scene_reflect", 0)
class ParticlesEngine:
        particle_id = 0

        def __init__(self, name, plus, scene, node_file_name, num_particles, start_scale, end_scale, stream_angle,color_label="teint"):
                self.name = name
                self.color_label=color_label
                self.particles_cnt = 0
                self.particles_cnt_f = 0
                self.num_particles = num_particles
                self.flow = 8
                self.particles_delay = 3
                self.particles = []
                self.create_particles(plus, scene, node_file_name)
                self.start_speed_range = hg.Vector2(800, 1200)
                self.delay_range = hg.Vector2(1, 2)
                self.start_scale = start_scale
                self.end_scale = end_scale
                self.scale_range = hg.Vector2(1,2)
                self.stream_angle = stream_angle
                self.colors = [hg.Color(1, 1, 1, 1), hg.Color(1, 1, 1, 0)]
                self.start_offset = 0
                self.rot_range_x = hg.Vector2(0, 0)
                self.rot_range_y = hg.Vector2(0, 0)
                self.rot_range_z = hg.Vector2(0, 0)
                self.gravity=hg.Vector3(0,-9.8,0)
                self.linear_damping = 1
                self.loop=True
                self.end=False #True when loop=True and all particles are dead
                self.num_new=0
                self.reset()

        def set_rot_range(self,xmin,xmax,ymin,ymax,zmin,zmax):
                self.rot_range_x = hg.Vector2(xmin, xmax)
                self.rot_range_y = hg.Vector2(ymin, ymax)
                self.rot_range_z = hg.Vector2(zmin, zmax)

        def create_particles(self, plus, scene, node_file_name):
                for i in range(self.num_particles):
                        node,geo = load_object(plus, node_file_name, self.name + "." + str(i), True)
                        particle = Particle(node)
                        scene.AddNode(particle.node)
                        self.particles.append(particle)

        def reset(self):
                self.num_new = 0
                self.particles_cnt = 0
                self.particles_cnt_f = 0
                self.end=False
                for i in range(self.num_particles):
                        self.particles[i].age = -1
                        self.particles[i].node.SetEnabled(False)
                        self.particles[i].v_move = hg.Vector3(0, 0, 0)

        def get_direction(self, main_dir):
                if self.stream_angle == 0: return main_dir
                axe0 = hg.Vector3(0, 0, 0)
                axeRot = hg.Vector3(0, 0, 0)
                while axeRot.Len() < 1e-4:
                        while axe0.Len() < 1e-5:
                                axe0 = hg.Vector3(uniform(-1, 1), uniform(-1, 1), uniform(-1, 1))
                        axe0.Normalize()
                        axeRot = hg.Cross(axe0, main_dir)
                axeRot.Normalize()
                return MathsSupp.rotate_vector(main_dir, axeRot, random() * radians(self.stream_angle))

        def update_color(self, particle: Particle):
                if len(self.colors) == 1:
                        c = self.colors[0]
                else:
                        c=MathsSupp.get_mix_color_value(particle.age / particle.delay,self.colors)
                particle.node.GetObject().GetGeometry().GetMaterial(0).SetFloat4(self.color_label, c.r, c.g, c.b, c.a)

        def update_kinetics(self, position: hg.Vector3, direction: hg.Vector3, v0: hg.Vector3, axisY: hg.Vector3, dts):
                self.num_new = 0
                if not self.end:
                        self.particles_cnt_f += dts * self.flow
                        self.num_new = int(self.particles_cnt_f) - self.particles_cnt
                        if self.num_new > 0:
                                for i in range(self.num_new):
                                        if not self.loop and self.particles_cnt+i>=self.num_particles:break
                                        particle = self.particles[(self.particles_cnt + i) % self.num_particles]
                                        particle.age = 0
                                        particle.delay = uniform(self.delay_range.x, self.delay_range.y)
                                        particle.scale = uniform(self.scale_range.x,self.scale_range.y)
                                        mat = particle.node.GetTransform()
                                        dir = self.get_direction(direction)
                                        rot_mat = hg.Matrix3(hg.Cross(axisY, dir), axisY, dir)
                                        mat.SetPosition(position + dir * self.start_offset)
                                        mat.SetRotationMatrix(rot_mat)
                                        mat.SetScale(self.start_scale)
                                        particle.rot_speed = hg.Vector3(uniform(self.rot_range_x.x, self.rot_range_x.y),
                                                                                                        uniform(self.rot_range_y.x, self.rot_range_y.y),
                                                                                                        uniform(self.rot_range_z.x, self.rot_range_z.y))
                                        particle.v_move = v0 + dir * uniform(self.start_speed_range.x, self.start_speed_range.y)
                                        particle.node.SetEnabled(False)
                                self.particles_cnt += self.num_new

                        n=0

                        for particle in self.particles:
                                if particle.age > particle.delay:
                                        particle.kill()
                                elif particle.age == 0:
                                        particle.age += dts
                                        n+=1
                                elif particle.age > 0:
                                        n+=1
                                        if not particle.node.GetEnabled(): particle.node.SetEnabled(True)
                                        t = particle.age / particle.delay
                                        mat = particle.node.GetTransform()
                                        pos = mat.GetPosition()
                                        rot = mat.GetRotation()
                                        particle.v_move += self.gravity * dts
                                        spd = particle.v_move.Len()
                                        particle.v_move -= particle.v_move.Normalized()*spd*self.linear_damping*dts
                                        pos += particle.v_move  * dts
                                        rot += particle.rot_speed * dts
                                        pos.y=max(0,pos.y)
                                        mat.SetPosition(pos)
                                        mat.SetRotation(rot)
                                        mat.SetScale((self.start_scale * (1 - t) + self.end_scale * t)*particle.scale)
                                        # material = particle.node.GetObject().GetGeometry().GetMaterial(0)
                                        # material.SetFloat4("self_color",1.,1.,0.,1-t)
                                        self.update_color(particle)
                                        # particle.node.GetObject().GetGeometry().GetMaterial(0).SetFloat4("teint", 1,1,1,1)
                                        particle.age += dts

                        if n==0 and not self.loop: self.end=True
class MathsSupp:
        @classmethod
        def rotate_vector(cls, point: hg.Vector3, axe: hg.Vector3, angle):
                axe.Normalize()
                dot_prod = point.x * axe.x + point.y * axe.y + point.z * axe.z
                cos_angle = cos(angle)
                sin_angle = sin(angle)

                return hg.Vector3(
                        cos_angle * point.x + sin_angle * (axe.y * point.z - axe.z * point.y) + (1 - cos_angle) * dot_prod * axe.x, \
                        cos_angle * point.y + sin_angle * (axe.z * point.x - axe.x * point.z) + (1 - cos_angle) * dot_prod * axe.y, \
                        cos_angle * point.z + sin_angle * (axe.x * point.y - axe.y * point.x) + (1 - cos_angle) * dot_prod * axe.z)

        @classmethod
        def rotate_matrix(cls, mat: hg.Matrix3, axe: hg.Vector3, angle):
                axeX = mat.GetX()
                axeY = mat.GetY()
                # axeZ=mat.GetZ()
                axeXr = cls.rotate_vector(axeX, axe, angle)
                axeYr = cls.rotate_vector(axeY, axe, angle)
                axeZr = hg.Cross(axeXr, axeYr)  # cls.rotate_vector(axeZ,axe,angle)
                return hg.Matrix3(axeXr, axeYr, axeZr)

        @classmethod
        def rotate_vector_2D(cls, p: hg.Vector2, angle):
                cos_angle = cos(angle)
                sin_angle = sin(angle)

                return hg.Vector2(p.x * cos_angle - p.y * sin_angle, p.x * sin_angle + p.y * cos_angle)

        @classmethod
        def get_sound_distance_level(cls, listener_position: hg.Vector3, sounder_position: hg.Vector3):
                distance = (sounder_position - listener_position).Len()
                return 1 / (distance / 10 + 1)


        @classmethod
        def get_mix_color_value(cls,f,colors):
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
class Destroyable_Machine:
        TYPE_SHIP = 1
        TYPE_AIRCRAFT = 2
        TYPE_MISSILE = 3
        TYPE_GROUND = 4

        def __init__(self, parent_node, type, nationality):
                self.type = type
                self.nationality = nationality
                self.parent_node = parent_node
                self.health_level = 1
                self.wreck = False
                self.activated = False  # Used by HUD radar
                self.v_move = hg.Vector3(0, 0, 0)
                self.ground_ray_cast_pos = hg.Vector3()
                self.ground_ray_cast_dir = hg.Vector3()
                self.ground_ray_cast_length = 2

        def get_parent_node(self):
                return self.parent_node

        def hit(self, value):
                raise NotImplementedError
class Missile(Destroyable_Machine):

        def __init__(self, name, nationality, plus, scene, audio, missile_file_name, smoke_file_name_prefix,
                                 smoke_color: hg.Color = hg.Color.White, start_position=hg.Vector3.Zero,
                                 start_rotation=hg.Vector3.Zero):
                self.name = name
                self.start_position = start_position
                self.start_rotation = start_rotation
                self.smoke_color = smoke_color
                self.gravity = hg.Vector3(0, -9.8, 0)
                self.audio = audio

                nd, geo = load_object(plus, missile_file_name, name, False)
                Destroyable_Machine.__init__(self, nd, Destroyable_Machine.TYPE_MISSILE, nationality)

                scene.AddNode(self.parent_node)
                self.smoke = []

                for i in range(17):
                        node, geo = load_object(plus, smoke_file_name_prefix + "." + str(i) + ".geo", name + ".smoke." + str(i),
                                                                        True)
                        scene.AddNode(node)
                        self.smoke.append(node)

                self.target = None
                self.f_thrust = 200
                self.moment_speed = 1.

                self.drag_coeff = hg.Vector3(0.37, 0.37, 0.0003)
                self.air_density = 1.225
                self.smoke_parts_distance = 1.44374
                self.set_smoke_solor(self.smoke_color)
                self.smoke_delay = 1
                self.smoke_time = 0

                self.life_delay = 20
                self.life_cptr = 0

                # Feed-backs:
                self.explode = ParticlesEngine(self.name + ".explode", plus, scene, "assets/feed_backs/feed_back_explode.geo",
                                                                           50, hg.Vector3(5, 5, 5), hg.Vector3(100, 100, 100), 180)
                self.explode.delay_range = hg.Vector2(1, 2)
                self.explode.flow = 0
                self.explode.scale_range = hg.Vector2(0.25, 2)
                self.explode.start_speed_range = hg.Vector2(0, 100)
                self.explode.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0., 0., 0.5), hg.Color(0., 0., 0., 0.25),
                                                           hg.Color(0., 0., 0., 0.125), hg.Color(0., 0., 0., 0.0)]
                self.explode.set_rot_range(radians(20), radians(50), radians(10), radians(45), radians(5), radians(15))
                self.explode.gravity = hg.Vector3(0, -9.8, 0)
                self.explode.loop = False

                # Sfx:
                self.explosion_settings = hg.MixerChannelState(0, 0, hg.MixerNoLoop)
                self.turbine_channel = None
                self.turbine_settings = hg.MixerChannelState(0, 0, hg.MixerRepeat)

        def hit(self, value):
                pass

        def reset(self, position=None, rotation=None):
                if self.turbine_channel is not None:
                        self.audio.Stop(self.turbine_channel)
                self.smoke_time = 0
                if position is not None:
                        self.start_position = position
                if rotation is not None:
                        self.start_rotation = rotation
                self.parent_node.GetTransform().SetPosition(self.start_position)
                self.parent_node.GetTransform().SetRotation(self.start_rotation)
                self.activated = False
                for node in self.smoke:
                        node.GetTransform().SetPosition(hg.Vector3(0, 0, 0))
                        node.SetEnabled(False)
                self.explode.reset()
                self.explode.flow = 0
                self.parent_node.SetEnabled(True)
                self.wreck = False
                self.v_move *= 0
                self.life_cptr = 0

        def set_smoke_solor(self, color: hg.Color):
                self.smoke_color = color
                for node in self.smoke:
                        node.GetTransform().SetPosition(hg.Vector3(0, 0, 0))
                        node.GetObject().GetGeometry().GetMaterial(0).SetFloat4("teint", self.smoke_color.r, self.smoke_color.g,
                                                                                                                                        self.smoke_color.b, self.smoke_color.a)

        def get_parent_node(self):
                return self.parent_node

        def start(self, target: Destroyable_Machine, v0: hg.Vector3):
                if not self.activated:
                        self.smoke_time = 0
                        self.life_cptr = 0
                        self.target = target
                        self.v_move = hg.Vector3(v0)
                        self.activated = True
                        pos = self.parent_node.GetTransform().GetPosition()
                        for node in self.smoke:
                                node.SetEnabled(True)
                                node.GetTransform().SetPosition(pos)

                        self.turbine_settings.volume = 0
                        self.turbine_channel = self.audio.Start(self.audio.LoadSound("assets/sfx/missile_engine.wav"),
                                                                                                        self.turbine_settings)

        def get_linear_speed(self):
                return self.v_move.Len()

        def update_smoke(self, target_point: hg.Vector3, dts):
                spd = self.get_linear_speed() * 0.033
                t = min(1, abs(self.smoke_time) / self.smoke_delay)
                self.smoke_time += dts
                n = len(self.smoke)
                color_end = self.smoke_color * t + hg.Color(1., 1., 1., 0.) * (1 - t)
                for i in range(n):
                        node = self.smoke[i]
                        ts = t * n
                        ti = int(ts)
                        if ti == i:
                                ts -= ti
                                alpha = ts * color_end.a
                        elif ti > i:
                                alpha = color_end.a
                        else:
                                alpha = 0
                        node.GetObject().GetGeometry().GetMaterial(0).SetFloat4("teint", color_end.r, color_end.g, color_end.b,
                                                                                                                                        alpha)

                        mat = node.GetTransform().GetWorld()
                        mat.SetScale(hg.Vector3(1, 1, 1))
                        pos = mat.GetTranslation()
                        v = target_point - pos
                        dir = v.Normalized()
                        # Position:
                        if v.Len() > self.smoke_parts_distance * spd:
                                pos = target_point - dir * self.smoke_parts_distance * spd * t
                                node.GetTransform().SetPosition(hg.Vector3(pos))  # node.SetEnabled(True)
                        # else:
                        # node.SetEnabled(False)
                        # Orientation:
                        aZ = mat.GetZ().Normalized()
                        axis_rot = hg.Cross(aZ, dir)
                        angle = axis_rot.Len()
                        if angle > 0.001:
                                # Rotation matrix:
                                ay = axis_rot.Normalized()
                                rot_mat = hg.Matrix3(hg.Cross(ay, dir), ay, dir)
                                node.GetTransform().SetRotationMatrix(rot_mat)
                        node.GetTransform().SetScale(hg.Vector3(1, 1, spd * t))
                        target_point = pos

        def update_kinetics(self, scene, dts):
                if self.activated:
                        self.life_cptr += dts
                        if self.life_cptr > self.life_delay:
                                self.start_explosion()
                        if not self.wreck:
                                mat = self.parent_node.GetTransform().GetWorld()
                                pos = mat.GetTranslation()

                                aX = mat.GetX()
                                aY = mat.GetY()
                                aZ = mat.GetZ()
                                # axis speed:
                                spdX = aX * hg.Dot(aX, self.v_move)
                                spdY = aY * hg.Dot(aY, self.v_move)
                                spdZ = aZ * hg.Dot(aZ, self.v_move)

                                q = hg.Vector3(pow(spdX.Len(), 2), pow(spdY.Len(), 2), pow(spdZ.Len(), 2)) * 0.5 * self.air_density

                                # Drag force:
                                F_drag = spdX.Normalized() * q.x * self.drag_coeff.x + spdY.Normalized() * q.y * self.drag_coeff.y + spdZ.Normalized() * q.z * self.drag_coeff.z

                                F_thrust = aZ * self.f_thrust

                                self.v_move += (F_thrust - F_drag + self.gravity) * dts

                                pos += self.v_move * dts
                                self.parent_node.GetTransform().SetPosition(pos)

                                # Rotation
                                if self.target is not None:
                                        target_node = self.target.get_parent_node()
                                        target_dir = (target_node.GetTransform().GetPosition() - pos).Normalized()
                                        axis_rot = hg.Cross(aZ, target_dir)
                                        if axis_rot.Len() > 0.001:
                                                # Rotation matrix:
                                                rot_mat = MathsSupp.rotate_matrix(mat.GetRotationMatrix(), axis_rot.Normalized(),
                                                                                                                  self.moment_speed * dts)
                                                self.parent_node.GetTransform().SetRotationMatrix(rot_mat)

                                # Collision
                                if self.target is not None:
                                        spd = self.v_move.Len()
                                        hit, impact = scene.GetPhysicSystem().Raycast(pos, self.v_move.Normalized(), 0x255, spd)
                                        if hit:
                                                if impact.GetNode() == target_node:
                                                        v_impact = impact.GetPosition() - pos
                                                        if v_impact.Len() < 2 * spd * dts:
                                                                self.start_explosion()
                                                                self.target.hit(0.35)
                                if pos.y < 0:
                                        self.start_explosion()
                                self.update_smoke(pos, dts)

                                level = MathsSupp.get_sound_distance_level(scene.GetCurrentCamera().GetTransform().GetPosition(),
                                                                                                                   self.parent_node.GetTransform().GetPosition())
                                self.turbine_settings.volume = level
                                self.audio.SetChannelState(self.turbine_channel, self.turbine_settings)

                        else:
                                pos = self.parent_node.GetTransform().GetPosition()
                                self.explode.update_kinetics(pos, hg.Vector3.Front, self.v_move, hg.Vector3.Up, dts)
                                if self.smoke_time < 0:
                                        self.update_smoke(pos, dts)
                                if self.explode.end and self.smoke_time >= 0:
                                        self.activated = False

        def start_explosion(self):
                if not self.wreck:
                        self.audio.Stop(self.turbine_channel)
                        self.explosion_settings.volume = min(1, self.turbine_settings.volume * 2)
                        self.audio.Start(self.audio.LoadSound("assets/sfx/missile_explosion.wav"), self.explosion_settings)
                        self.wreck = True
                        self.explode.flow = 3000
                        self.parent_node.SetEnabled(False)
                        self.smoke_time = -self.smoke_delay
class MachineGun(ParticlesEngine):
        def __init__(self, name, plus, scene):
                ParticlesEngine.__init__(self, name, plus, scene, "assets/weaponry/gun_bullet.geo", 24, hg.Vector3(2, 2, 20),
                                                                 hg.Vector3(20, 20, 100), 0.1, "self_color")

                self.start_speed_range = hg.Vector2(2000, 2000)
                self.delay_range = hg.Vector2(2, 2)
                self.start_offset = 0  # self.start_scale.z
                self.gravity = hg.Vector3.Zero
                self.linear_damping = 0
                self.scale_range = hg.Vector2(1, 1)

                self.bullets_feed_backs = []
                for i in range(self.num_particles):
                        fb = ParticlesEngine(self.name + ".fb." + str(i), plus, scene, "assets/feed_backs/bullet_impact.geo", 5,
                                                                 hg.Vector3(1, 1, 1), hg.Vector3(10, 10, 10), 180)
                        fb.delay_range = hg.Vector2(1, 1)
                        fb.flow = 0
                        fb.scale_range = hg.Vector2(1, 3)
                        fb.start_speed_range = hg.Vector2(0, 20)
                        fb.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., .5, 0.25, 0.25), hg.Color(0.1, 0., 0., 0.)]
                        fb.set_rot_range(radians(20), radians(50), radians(10), radians(45), radians(5), radians(15))
                        fb.gravity = hg.Vector3(0, 0, 0)
                        fb.loop = False
                        self.bullets_feed_backs.append(fb)

        def strike(self, i):
                self.particles[i].kill()
                fb = self.bullets_feed_backs[i]
                fb.reset()
                fb.flow = 3000

        def update_kinetics(self, scene, targets, position: hg.Vector3, direction: hg.Vector3, v0: hg.Vector3,
                                                axisY: hg.Vector3, dts):
                ParticlesEngine.update_kinetics(self, position, direction, v0, axisY, dts)
                for i in range(self.num_particles):
                        bullet = self.particles[i]
                        mat = bullet.node.GetTransform().GetWorld()
                        pos_fb = mat.GetTranslation()
                        pos = mat.GetTranslation() - mat.GetZ()

                        if bullet.node.GetEnabled():
                                spd = bullet.v_move.Len()
                                if pos_fb.y < 1:
                                        bullet.v_move *= 0
                                        self.strike(i)
                                hit, impact = scene.GetPhysicSystem().Raycast(pos, bullet.v_move.Normalized(), 0x255, spd)
                                if hit:
                                        if (impact.GetPosition() - pos).Len() < spd * dts * 2:
                                                for target in targets:
                                                        if target.get_parent_node() == impact.GetNode():
                                                                target.hit(0.1)
                                                                bullet.v_move = target.v_move
                                                                self.strike(i)
                        fb = self.bullets_feed_backs[i]
                        if not fb.end and fb.flow > 0:
                                fb.update_kinetics(pos_fb, hg.Vector3.Front, bullet.v_move, hg.Vector3.Up, dts)
class Aircraft(Destroyable_Machine):
        main_node = None

        def __init__(self, name, nationality, id_string, plus, scene, start_position: hg.Vector3,
                                 start_rotation=hg.Vector3.Zero):
                self.name = name
                self.id_string = id_string

                Destroyable_Machine.__init__(self, scene.GetNode("dummy_" + id_string + "_fuselage"),
                                                                         Destroyable_Machine.TYPE_AIRCRAFT, nationality)
                self.wing_l = scene.GetNode("dummy_" + id_string + "_configurable_wing_l")
                self.wing_r = scene.GetNode("dummy_" + id_string + "_configurable_wing_r")
                self.aileron_l = scene.GetNode("dummy_" + id_string + "_aileron_l")
                self.aileron_r = scene.GetNode("dummy_" + id_string + "_aileron_r")
                self.elevator_l = scene.GetNode("dummy_" + id_string + "_elevator_changepitch_l")
                self.elevator_r = scene.GetNode("dummy_" + id_string + "_elevator_changepitch_r")
                self.rudder_l = scene.GetNode(id_string + "_rudder_changeyaw_l")
                self.rudder_r = scene.GetNode(id_string + "_rudder_changeyaw_r")

                self.wings_max_angle = 45
                self.wings_level = 0
                self.wings_thresholds = hg.Vector2(500, 750)
                self.wings_geometry_gain_friction = -0.0001

                for nd in scene.GetNodes():
                        if nd.GetName().split("_")[0] == "dummy":
                                nd.RemoveComponent(nd.GetObject())

                self.start_position = start_position
                self.start_rotation = start_rotation
                self.thrust_level = 0
                self.thrust_force = 10
                self.post_combution = False
                self.post_combution_force = self.thrust_force / 2

                self.angular_frictions = hg.Vector3(0.000175, 0.000125, 0.000275)  # pitch, yaw, roll
                self.speed_ceiling = 1750  # maneuverability is not guaranteed beyond this speed !
                self.angular_levels = hg.Vector3(0, 0, 0)  # 0 to 1
                self.angular_levels_dest = hg.Vector3(0, 0, 0)
                self.angular_levels_inertias = hg.Vector3(3, 3, 3)
                self.parts_angles = hg.Vector3(radians(15), radians(45), radians(45))
                self.angular_speed = hg.Vector3(0, 0, 0)

                self.drag_coeff = hg.Vector3(0.033, 0.06666, 0.0002)
                self.wings_lift = 0.0005
                self.brake_level = 0
                self.brake_drag = 0.006
                self.flaps_level = 0
                self.flaps_lift = 0.0025
                self.flaps_drag = 0.002

                self.flag_easy_steering = True

                # Physic parameters:
                self.F_gravity = hg.Vector3(0, -9.8, 0)
                self.air_density = 1.225  # Pour plus de réalisme, à modifier en fonction de l'altitude

                # collisions:
                self.rigid, self.rigid_wing_r, self.rigid_wing_l, self.collisions = self.get_collisions(scene)

                self.flag_landed = True

                # Missiles:
                self.missiles_slots = self.get_missiles_slots(scene)
                self.missiles = [None] * 4
                self.missiles_started = []

                self.targets = []
                self.target_id = 0
                self.target_lock_range = hg.Vector2(100, 3000)  # Target lock distance range
                self.target_lock_delay = hg.Vector2(1, 5)  # Target lock delay in lock range
                self.target_lock_t = 0
                self.target_locking_state = 0  # 0 to 1
                self.target_locked = False
                self.target_out_of_range = False
                self.target_distance = 0
                self.target_cap = 0
                self.target_altitude = 0
                self.target_angle = 0

                # Gun machine:
                self.gun_position = hg.Vector3(0, -0.65, 9.8)
                self.gun_machine = MachineGun(self.name + ".gun", plus, scene)

                # Feed-backs:
                self.explode = ParticlesEngine(self.name + ".explode", plus, scene, "assets/feed_backs/feed_back_explode.geo",
                                                                           100, hg.Vector3(10, 10, 10), hg.Vector3(100, 100, 100), 180)
                self.explode.delay_range = hg.Vector2(1, 4)
                self.explode.flow = 0
                self.explode.scale_range = hg.Vector2(0.25, 2)
                self.explode.start_speed_range = hg.Vector2(0, 100)
                self.explode.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0., 0., 0.5), hg.Color(.5, .5, .5, 0.25),
                                                           hg.Color(0., 0., 0., 0.125), hg.Color(0., 0., 0., 0.0)]
                self.explode.set_rot_range(radians(20), radians(50), radians(10), radians(45), radians(5), radians(15))
                self.explode.gravity = hg.Vector3(0, -9.8, 0)
                self.explode.loop = False

                self.smoke = ParticlesEngine(self.name + ".smoke", plus, scene, "assets/feed_backs/feed_back_explode.geo", 400,
                                                                         hg.Vector3(5, 5, 5), hg.Vector3(50, 50, 50), 180)
                self.smoke.delay_range = hg.Vector2(4, 8)
                self.smoke.flow = 0
                self.smoke.scale_range = hg.Vector2(0.1, 5)
                self.smoke.start_speed_range = hg.Vector2(5, 15)
                self.smoke.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0., 0., 0.3), hg.Color(.7, .7, .7, 0.2),
                                                         hg.Color(.0, .0, .0, 0.1), hg.Color(0., 0., 0., 0.05), hg.Color(0., 0., 0., 0)]
                self.smoke.set_rot_range(0, 0, radians(120), radians(120), 0, 0)
                self.smoke.gravity = hg.Vector3(0, 30, 0)
                self.smoke.linear_damping = 0.5
                self.smoke.loop = True

                self.engines_position = [hg.Vector3(1.56887, -0.14824, -6.8), hg.Vector3(-1.56887, -0.14824, -6.8)]
                self.pc_r = self.create_post_combustion_particles(plus, scene, ".pcr")
                self.pc_l = self.create_post_combustion_particles(plus, scene, ".pcl")

                self.destroyable_targets = []

                # Linear acceleration:
                self.linear_acceleration = 0
                self.linear_speeds = [0] * 10
                self.linear_spd_rec_cnt = 0

                # Attitudes calculation:
                self.horizontal_aX = None
                self.horizontal_aY = None
                self.horizontal_aZ = None
                self.pitch_attitude = 0
                self.roll_attitude = 0
                self.y_dir = 1

                self.cap = 0

                # IA
                self.IA_activated = False
                self.IA_fire_missiles_delay = 10
                self.IA_fire_missile_cptr = 0
                self.IA_flag_altitude_correction = False
                self.IA_altitude_min = 500
                self.IA_altitude_safe = 2000
                self.IA_gun_distance_max = 1000
                self.IA_gun_angle = 10
                self.IA_cruising_altitude = 3000

                # Autopilot (used by IA)
                self.autopilot_activated = False
                self.autopilot_altitude = 1000
                self.autopilot_cap = 0
                self.autopilot_pitch_attitude = 0
                self.autopilot_roll_attitude = 0

                self.reset()

        def create_post_combustion_particles(self, plus, scene, engine_name):
                pc = ParticlesEngine(self.name + engine_name, plus, scene, "assets/feed_backs/bullet_impact.geo", 15,
                                                         hg.Vector3(1, 1, 1), hg.Vector3(0.2, 0.2, 0.2), 15)
                pc.delay_range = hg.Vector2(0.3, 0.4)
                pc.flow = 0
                pc.scale_range = hg.Vector2(1, 1)
                pc.start_speed_range = hg.Vector2(1, 1)
                pc.colors = [hg.Color(1., 1., 1., 1), hg.Color(1., 0.9, 0.7, 0.5), hg.Color(0.9, 0.7, 0.1, 0.25),
                                         hg.Color(0.9, 0.5, 0., 0.), hg.Color(0.85, 0.5, 0., 0.25), hg.Color(0.8, 0.4, 0., 0.15),
                                         hg.Color(0.8, 0.1, 0.1, 0.05), hg.Color(0.5, 0., 0., 0.)]
                pc.set_rot_range(radians(1200), radians(2200), radians(1420), radians(1520), radians(1123), radians(5120))
                pc.gravity = hg.Vector3(0, 0, 0)
                pc.linear_damping = 0
                pc.loop = True
                return pc

        def set_destroyable_targets(self, targets):
                self.destroyable_targets = targets

        def hit(self, value):
                if not self.wreck:
                        self.set_health_level(self.health_level - value)
                        if self.health_level == 0 and not self.wreck:
                                self.start_explosion()
                                self.set_thrust_level(0)

        def get_missiles_slots(self, scene):
                slots = []
                # slots=[hg.Vector3(0,0,0),hg.Vector3(0,0,0),hg.Vector3(0,0,0),hg.Vector3(0,0,0)]
                # return slots
                for i in range(4):
                        nd = scene.GetNode("dummy_" + self.id_string + "_slot." + str(i + 1))
                        slots.append(nd.GetTransform().GetPosition())  # scene.RemoveNode(nd)
                return slots

        def get_collisions(self, scene):
                rigid, rigid_wing_r, rigid_wing_l = hg.RigidBody(), hg.RigidBody(), hg.RigidBody()
                rigid.SetType(hg.RigidBodyKinematic)
                rigid_wing_r.SetType(hg.RigidBodyKinematic)
                rigid_wing_l.SetType(hg.RigidBodyKinematic)
                self.parent_node.AddComponent(rigid)
                self.wing_l.AddComponent(rigid_wing_l)
                self.wing_r.AddComponent(rigid_wing_r)
                collisions_nodes = []
                for nd in scene.GetNodes():
                        if nd.GetName().find(self.id_string + "_col_shape") >= 0:
                                collisions_nodes.append(nd)
                collisions_boxes = []
                for col_shape in collisions_nodes:
                        colbox = hg.BoxCollision()
                        collisions_boxes.append(colbox)
                        obj = col_shape.GetObject()
                        bounds = obj.GetLocalMinMax()
                        dimensions = bounds.mx - bounds.mn
                        pos = col_shape.GetTransform().GetPosition() + bounds.mn + dimensions * 0.5
                        colbox.SetDimensions(dimensions)
                        colbox.SetMatrix(hg.Matrix4.TranslationMatrix(pos))
                        if col_shape.GetName().find("wing_l") >= 0:
                                self.wing_l.AddComponent(colbox)
                        elif col_shape.GetName().find("wing_r") >= 0:
                                self.wing_r.AddComponent(colbox)
                        else:
                                self.parent_node.AddComponent(colbox)
                        scene.RemoveNode(col_shape)
                return rigid, rigid_wing_r, rigid_wing_l, collisions_boxes

        def reset(self, position=None, rotation=None):
                if position is not None:
                        self.start_position = position
                if rotation is not None:
                        self.start_rotation = rotation
                self.parent_node.GetTransform().SetPosition(self.start_position)
                self.parent_node.GetTransform().SetRotation(self.start_rotation)

                self.v_move = hg.Vector3(0, 0, 0)
                self.angular_levels = hg.Vector3(0, 0, 0)
                self.set_thrust_level(0)
                self.deactivate_post_combution()
                self.flaps_level = 0
                self.brake_level = 0

                self.missiles = [None] * 4
                self.missiles_started = []

                if self.gun_machine is not None:
                        self.gun_machine.reset()
                        self.gun_machine.flow = 0

                self.smoke.reset()
                self.explode.reset()
                self.wreck = False
                self.activated = True
                self.explode.flow = 0
                self.set_health_level(1)
                self.target_id = 0
                self.target_lock_t = 0
                self.target_locked = False
                self.target_out_of_range = False
                self.target_locking_state = 0

                self.linear_speeds = [0] * 10

        def get_world_speed(self):
                sX = hg.Vector3.Right * (hg.Dot(hg.Vector3.Right, self.v_move))
                sZ = hg.Vector3.Front * (hg.Dot(hg.Vector3.Front, self.v_move))
                vs = hg.Dot(hg.Vector3.Up, self.v_move)
                hs = (sX + sZ).Len()
                return hs, vs

        def set_linear_speed(self, value):
                aZ = self.parent_node.GetTransform().GetWorld().GetZ()
                self.v_move = aZ * value

        def get_linear_speed(self):
                return self.v_move.Len()

        def get_altitude(self):
                return self.parent_node.GetTransform().GetPosition().y

        def set_thrust_level(self, value):
                self.thrust_level = min(max(value, 0), 1)
                if self.thrust_level < 1: self.deactivate_post_combution()

        def set_health_level(self, value):
                self.health_level = min(max(value, 0), 1)
                if self.health_level < 1:
                        self.smoke.flow = 40
                else:
                        self.smoke.flow = 0
                self.smoke.delay_range = hg.Vector2(1, 10) * pow(1 - self.health_level, 3)
                self.smoke.scale_range = hg.Vector2(0.1, 5) * pow(1 - self.health_level, 3)
                self.smoke.stream_angle = pow(1 - self.health_level, 2.6) * 180

        def start_explosion(self):
                self.wreck = True
                self.explode.flow = 500

        def set_wings_level(self, value):
                self.wings_level = min(max(value, 0), 1)
                rot = self.wing_l.GetTransform().GetRotation()
                rot.y = -radians(self.wings_max_angle * self.wings_level)
                self.wing_l.GetTransform().SetRotation(rot)

                rot = self.wing_r.GetTransform().GetRotation()
                rot.y = radians(self.wings_max_angle * self.wings_level)
                self.wing_r.GetTransform().SetRotation(rot)

        def set_brake_level(self, value):
                self.brake_level = min(max(value, 0), 1)

        def set_flaps_level(self, value):
                self.flaps_level = min(max(value, 0), 1)

        def set_pitch_level(self, value):
                self.angular_levels_dest.x = max(min(1, value), -1)

        def set_yaw_level(self, value):
                self.angular_levels_dest.y = max(min(1, value), -1)

        def set_roll_level(self, value):
                self.angular_levels_dest.z = max(min(1, value), -1)

        def set_autopilot_pitch_attitude(self, value):
                self.autopilot_pitch_attitude = max(min(180, value), -180)

        def set_autopilot_roll_attitude(self, value):
                self.autopilot_roll_attitude = max(min(180, value), -180)

        def set_autopilot_cap(self, value):
                self.autopilot_cap = max(min(360, value), 0)

        def set_autopilot_altitude(self, value):
                self.autopilot_altitude = value

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

        def update_angular_levels(self, dts):
                self.angular_levels.x = self.update_inertial_value(self.angular_levels.x, self.angular_levels_dest.x,
                                                                                                                   self.angular_levels_inertias.x, dts)
                self.angular_levels.y = self.update_inertial_value(self.angular_levels.y, self.angular_levels_dest.y,
                                                                                                                   self.angular_levels_inertias.y, dts)
                self.angular_levels.z = self.update_inertial_value(self.angular_levels.z, self.angular_levels_dest.z,
                                                                                                                   self.angular_levels_inertias.z, dts)

        def stabilize(self, dts, p, y, r):
                if p: self.set_pitch_level(0)
                if y: self.set_yaw_level(0)
                if r: self.set_roll_level(0)

        def activate_post_combution(self):
                if self.thrust_level == 1:
                        self.post_combution = True
                        self.pc_r.flow = 35
                        self.pc_l.flow = 35

        def deactivate_post_combution(self):
                self.post_combution = False
                self.pc_r.flow = 0
                self.pc_l.flow = 0

        def fire_gun_machine(self):
                if not self.wreck:
                        self.gun_machine.flow = 24 / 2

        def stop_gun_machine(self):
                self.gun_machine.flow = 0

        def is_gun_activated(self):
                if self.gun_machine.flow == 0:
                        return False
                else:
                        return True

        def update_mobile_parts(self):
                self.elevator_l.GetTransform().SetRotation(hg.Vector3(-self.parts_angles.x * self.angular_levels.x, 0, 0))
                self.elevator_r.GetTransform().SetRotation(hg.Vector3(-self.parts_angles.x * self.angular_levels.x, 0, 0))

                rot_l, rot_r = self.rudder_l.GetTransform().GetRotation(), self.rudder_r.GetTransform().GetRotation()
                rot_l.y = self.parts_angles.y * self.angular_levels.y + pi
                rot_r.y = -self.parts_angles.y * self.angular_levels.y
                self.rudder_l.GetTransform().SetRotation(rot_l)
                self.rudder_r.GetTransform().SetRotation(rot_r)

                rot_l, rot_r = self.aileron_l.GetTransform().GetRotation(), self.aileron_r.GetTransform().GetRotation()
                rot_l.x = -self.parts_angles.z * self.angular_levels.z
                rot_r.x = -self.parts_angles.z * self.angular_levels.z
                self.aileron_l.GetTransform().SetRotation(rot_l)
                self.aileron_r.GetTransform().SetRotation(rot_r)

        def rec_linear_speed(self):
                self.linear_speeds[self.linear_spd_rec_cnt] = self.v_move.Len()
                self.linear_spd_rec_cnt += 1
                if self.linear_spd_rec_cnt >= len(self.linear_speeds):
                        self.linear_spd_rec_cnt = 0

        def update_linear_acceleration(self):
                m = 0
                for s in self.linear_speeds:
                        m += s
                m /= len(self.linear_speeds)
                self.linear_acceleration = self.v_move.Len() - m

        def get_linear_acceleration(self):
                return self.linear_acceleration

        def update_post_combustion_particles(self, dts, pos, rot_mat):
                self.pc_r.update_kinetics(self.engines_position[0] * rot_mat + pos, rot_mat.GetZ() * -1, self.v_move,
                                                                  rot_mat.GetY(), dts)
                self.pc_l.update_kinetics(self.engines_position[1] * rot_mat + pos, rot_mat.GetZ() * -1, self.v_move,
                                                                  rot_mat.GetY(), dts)

        def update_IA(self, dts):
                alt = self.get_altitude()
                if self.target_id > 0:
                        self.set_autopilot_cap(self.target_cap)
                        if self.IA_flag_altitude_correction:
                                self.set_autopilot_altitude(self.IA_altitude_safe)
                                if self.IA_altitude_safe - 100 < alt < self.IA_altitude_safe + 100:
                                        self.IA_flag_altitude_correction = False
                        else:
                                self.set_autopilot_altitude(self.target_altitude)
                                if alt < self.IA_altitude_min:
                                        self.IA_flag_altitude_correction = True

                        if self.target_locked:
                                if self.IA_fire_missile_cptr <= 0:
                                        self.fire_missile()
                                        self.IA_fire_missile_cptr = self.IA_fire_missiles_delay
                                if self.IA_fire_missile_cptr > 0:
                                        self.IA_fire_missile_cptr -= dts

                        if self.target_angle < self.IA_gun_angle and self.target_distance < self.IA_gun_distance_max:
                                self.fire_gun_machine()
                        else:
                                self.stop_gun_machine()

                else:
                        self.set_autopilot_altitude(self.IA_cruising_altitude)
                        self.set_autopilot_cap(0)
                        self.stop_gun_machine()

                if self.pitch_attitude > 15:
                        self.set_thrust_level(1)
                        self.activate_post_combution()
                elif -15 < self.pitch_attitude < 15:
                        self.deactivate_post_combution()
                        self.set_thrust_level(1)
                else:
                        self.deactivate_post_combution()
                        self.set_thrust_level(0.5)

        def update_autopilot(self, dts):
                # straighten aircraft:
                if self.y_dir < 0:
                        self.set_roll_level(0)
                        self.set_pitch_level(0)
                        self.set_yaw_level(0)
                else:
                        # cap / roll_attitude:
                        diff = self.autopilot_cap - self.cap
                        if diff > 180:
                                diff -= 360
                        elif diff < -180:
                                diff += 360

                        tc = max(-1, min(1, -diff / 90))
                        if tc < 0:
                                tc = -pow(-tc, 0.5)
                        else:
                                tc = pow(tc, 0.5)
                        self.set_autopilot_roll_attitude(tc * 85)

                        diff = self.autopilot_roll_attitude - self.roll_attitude
                        tr = max(-1, min(1, diff / 20))
                        self.set_roll_level(tr)

                        # altitude / pitch_attitude:
                        diff = self.autopilot_altitude - self.get_altitude()
                        ta = max(-1, min(1, diff / 500))

                        if ta < 0:
                                ta = -pow(-ta, 0.7)
                        else:
                                ta = pow(ta, 0.7)

                        self.set_autopilot_pitch_attitude(ta * 45)

                        diff = self.autopilot_pitch_attitude - self.pitch_attitude
                        tp = max(-1, min(1, diff / 10))
                        self.set_pitch_level(-tp)

        def calculate_cap(self, h_dir: hg.Vector3):
                cap = degrees(acos(max(-1, min(1, hg.Dot(h_dir, hg.Vector3.Front)))))
                if h_dir.x < 0: cap = 360 - cap
                return cap

        def update_kinetics(self, scene, dts):
                if self.activated:
                        #                               AJOUTER UNE CONDITION SI WRECK = TRUE (CRASH EN PIQUE + ROTATION AXEZ)
                        #                               Meshe de substitution ?

                        if self.IA_activated:
                                self.update_IA(dts)
                        if self.autopilot_activated or self.IA_activated:
                                self.update_autopilot(dts)

                        self.update_angular_levels(dts)
                        self.update_mobile_parts()

                        mat = self.parent_node.GetTransform().GetWorld()
                        aX = mat.GetX()
                        aY = mat.GetY()
                        aZ = mat.GetZ()

                        # Cap, Pitch & Roll attitude:
                        if aY.y > 0:
                                self.y_dir = 1
                        else:
                                self.y_dir = -1

                        self.horizontal_aZ = hg.Vector3(aZ.x, 0, aZ.z).Normalized()
                        self.horizontal_aX = hg.Cross(hg.Vector3.Up, self.horizontal_aZ) * self.y_dir
                        self.horizontal_aY = hg.Cross(aZ, self.horizontal_aX)  # ! It's not an orthogonal repere !

                        self.pitch_attitude = degrees(acos(max(-1, min(1, hg.Dot(self.horizontal_aZ, aZ)))))
                        if aZ.y < 0: self.pitch_attitude *= -1

                        self.roll_attitude = degrees(acos(max(-1, min(1, hg.Dot(self.horizontal_aX, aX)))))
                        if aX.y < 0: self.roll_attitude *= -1

                        self.cap = self.calculate_cap(self.horizontal_aZ)

                        # axis speed:
                        spdX = aX * hg.Dot(aX, self.v_move)
                        spdY = aY * hg.Dot(aY, self.v_move)
                        spdZ = aZ * hg.Dot(aZ, self.v_move)

                        frontal_speed = spdZ.Len()

                        # wings_geometry:
                        self.set_wings_level(max(min(
                                (frontal_speed * 3.6 - self.wings_thresholds.x) / (self.wings_thresholds.y - self.wings_thresholds.x),
                                1), 0))

                        # Thrust force:
                        k = pow(self.thrust_level, 2) * self.thrust_force
                        if self.post_combution and self.thrust_level == 1:
                                k += self.post_combution_force
                        F_thrust = mat.GetZ() * k

                        # Dynamic pressure:
                        q = hg.Vector3(pow(spdX.Len(), 2), pow(spdY.Len(), 2), pow(spdZ.Len(), 2)) * 0.5 * self.air_density

                        # F Lift
                        F_lift = aY * q.z * (self.wings_lift + self.flaps_level * self.flaps_lift)

                        # Drag force:
                        F_drag = spdX.Normalized() * q.x * self.drag_coeff.x + spdY.Normalized() * q.y * self.drag_coeff.y + spdZ.Normalized() * q.z * (
                                        self.drag_coeff.z + self.brake_drag * self.brake_level + self.flaps_level * self.flaps_drag + self.wings_geometry_gain_friction * self.wings_level)

                        # Total
                        self.v_move += ((F_thrust + F_lift - F_drag) * self.health_level + self.F_gravity) * dts

                        # Displacement:
                        pos = mat.GetTranslation()
                        pos += self.v_move * dts

                        # Rotations:
                        F_pitch = self.angular_levels.x * q.z * self.angular_frictions.x
                        F_yaw = self.angular_levels.y * q.z * self.angular_frictions.y
                        F_roll = self.angular_levels.z * q.z * self.angular_frictions.z

                        # Angular damping:
                        gaussian = exp(-pow(frontal_speed * 3.6 * 3 / self.speed_ceiling, 2) / 2)

                        # Angular speed:
                        self.angular_speed = hg.Vector3(F_pitch, F_yaw, F_roll) * gaussian

                        # Moment:
                        pitch_m = aX * self.angular_speed.x
                        yaw_m = aY * self.angular_speed.y
                        roll_m = aZ * self.angular_speed.z

                        # Easy steering:
                        if self.flag_easy_steering or self.autopilot_activated:

                                easy_yaw_angle = (1 - (hg.Dot(aX, self.horizontal_aX)))
                                if hg.Dot(aZ, hg.Cross(aX, self.horizontal_aX)) < 0:
                                        easy_turn_m_yaw = self.horizontal_aY * -easy_yaw_angle
                                else:
                                        easy_turn_m_yaw = self.horizontal_aY * easy_yaw_angle

                                easy_roll_stab = hg.Cross(aY, self.horizontal_aY) * self.y_dir
                                if self.y_dir < 0:
                                        easy_roll_stab.Normalize()
                                else:
                                        n = easy_roll_stab.Len()
                                        if n > 0.1:
                                                easy_roll_stab.Normalize()
                                                easy_roll_stab *= (1 - n) * n + n * pow(n, 0.125)

                                zl = min(1, abs(self.angular_levels.z + self.angular_levels.x + self.angular_levels.y))
                                roll_m += (easy_roll_stab * (1 - zl) + easy_turn_m_yaw) * q.z * self.angular_frictions.y * gaussian

                        # Moment:
                        moment = yaw_m + roll_m + pitch_m
                        axis_rot = moment.Normalized()
                        moment_speed = moment.Len() * self.health_level

                        # Rotation matrix:
                        rot_mat = MathsSupp.rotate_matrix(mat.GetRotationMatrix(), axis_rot, moment_speed * dts)
                        self.parent_node.GetTransform().SetRotationMatrix(rot_mat)

                        # Ground collisions:
                        self.flag_landed = False
                        self.ground_ray_cast_pos = pos - aY
                        self.ground_ray_cast_dir = aY * -1
                        hit, impact = scene.GetPhysicSystem().Raycast(self.ground_ray_cast_pos, self.ground_ray_cast_dir, 0x255,
                                                                                                                  self.ground_ray_cast_length)
                        if hit and impact.GetNode() != self.parent_node:
                                i_pos = impact.GetPosition()
                                alt = i_pos.y + 2
                        else:
                                alt = 4
                        if pos.y < alt:
                                if degrees(abs(asin(aZ.y))) < 15 and degrees(abs(asin(aX.y))) < 10 and frontal_speed * 3.6 < 300:

                                        pos.y += (alt - pos.y) * 0.1 * 60 * dts
                                        if self.v_move.y < 0: self.v_move.y *= pow(0.8, 60 * dts)
                                        b = min(1, self.brake_level + (1 - self.health_level))
                                        self.v_move *= ((b * pow(0.8, 60 * dts)) + (1 - b))
                                        self.flag_landed = True
                                else:
                                        pos.y = alt
                                        self.hit(1)
                                        self.v_move *= pow(0.9, 60 * dts)

                        self.parent_node.GetTransform().SetPosition(pos)

                        # Gun:
                        self.gun_machine.update_kinetics(scene, self.destroyable_targets, rot_mat * self.gun_position + pos, aZ,
                                                                                         self.v_move, aY, dts)
                        # Missiles:
                        self.update_target_lock(dts)
                        for missile in self.missiles_started:
                                missile.update_kinetics(scene, dts)

                        # Feed backs:
                        if self.health_level < 1:
                                self.smoke.update_kinetics(pos, aZ * -1, self.v_move, aY,
                                                                                   dts)  # AJOUTER UNE DUREE LIMITE AU FOURNEAU LORSQUE WRECK=TRUE !
                        if self.wreck and not self.explode.end:
                                self.explode.update_kinetics(pos, aZ * -1, self.v_move, aY, dts)

                        self.update_post_combustion_particles(dts, pos, rot_mat)

                        self.rec_linear_speed()
                        self.update_linear_acceleration()

        def update_target_lock(self, dts):
                if self.target_id > 0:
                        target = self.targets[self.target_id - 1]
                        if target.wreck or not target.activated:
                                self.next_target()
                                if self.target_id == 0:
                                        return
                        t_pos = self.targets[self.target_id - 1].get_parent_node().GetTransform().GetPosition()
                        mat = self.parent_node.GetTransform().GetWorld()
                        dir = mat.GetZ()
                        v = t_pos - mat.GetTranslation()
                        self.target_cap = self.calculate_cap((v * hg.Vector3(1, 0, 1)).Normalized())
                        self.target_altitude = t_pos.y
                        self.target_distance = v.Len()
                        t_dir = v.Normalized()
                        self.target_angle = degrees(acos(max(-1, min(1, hg.Dot(dir, t_dir)))))
                        if self.target_angle < 15 and self.target_lock_range.x < self.target_distance < self.target_lock_range.y:
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

        def fit_missile(self, missile: Missile, slot_id):
                nd = missile.get_parent_node()
                nd.GetTransform().SetParent(self.parent_node)
                missile.reset(self.missiles_slots[slot_id], hg.Vector3(0, 0, 0))
                self.missiles[slot_id] = missile

        def set_target_id(self, id):
                self.target_id = id
                if id > 0:
                        if self.targets is None or len(self.targets) == 0:
                                self.target_id = 0
                        target = self.targets[id - 1]
                        if target.wreck or not target.activated:
                                self.next_target()

        def next_target(self):
                if self.targets is not None:
                        self.target_locked = False
                        self.target_lock_t = 0
                        self.target_locking_state = 0
                        self.target_id += 1
                        if self.target_id > len(self.targets):
                                self.target_id = 0
                                return
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

        def get_target(self):
                if self.target_id > 0:
                        return self.targets[self.target_id - 1]
                else:
                        return None

        def fire_missile(self):
                if not self.wreck:
                        for i in range(len(self.missiles)):
                                missile = self.missiles[i]
                                if missile is not None:
                                        self.missiles[i] = None
                                        trans = missile.get_parent_node().GetTransform()
                                        mat = trans.GetWorld()
                                        trans.SetParent(Aircraft.main_node)
                                        trans.SetWorld(mat)
                                        if self.target_locked:
                                                target = self.targets[self.target_id - 1]
                                        else:
                                                target = None
                                        missile.start(target, self.v_move)
                                        self.missiles_started.append(missile)
                                        break


class AircraftSFX:
        def __init__(self, aircraft: Aircraft):
                self.aircraft = aircraft

                self.turbine_pitch_levels = hg.Vector2(1, 2)
                self.turbine_settings = hg.MixerChannelState(0, 0, hg.MixerRepeat)
                self.air_settings = hg.MixerChannelState(0, 0, hg.MixerRepeat)
                self.pc_settings = hg.MixerChannelState(0, 0, hg.MixerRepeat)
                self.wind_settings = hg.MixerChannelState(0, 0, hg.MixerRepeat)
                self.explosion_settings = hg.MixerChannelState(0, 0, hg.MixerNoLoop)
                self.machine_gun_settings = hg.MixerChannelState(0, 0, hg.MixerNoLoop)
                self.start = False

                self.pc_cptr = 0
                self.pc_start_delay = 0.25
                self.pc_stop_delay = 0.5

                self.turbine_chan = 0
                self.wind_chan = 0
                self.air_chan = 0
                self.pc_chan = 0
                self.pc_started = False
                self.pc_stopped = False

                self.exploded = False

        def reset(self):
                self.exploded = False

        def set_air_pitch(self, value):
                self.air_settings.pitch = value

        def set_pc_pitch(self, value):
                self.pc_settings.pitch = value

        def set_turbine_pitch_levels(self, values: hg.Vector2):
                self.turbine_pitch_levels = values

        def start_engine(self, main):
                self.turbine_settings.volume = 0
                self.turbine_settings.pitch = 1
                self.air_settings.volume = 0
                self.pc_settings.volume = 0
                self.air_chan = main.audio.Start(main.audio.LoadSound("assets/sfx/air.wav"), self.air_settings)
                self.turbine_chan = main.audio.Start(main.audio.LoadSound("assets/sfx/turbine.wav"), self.turbine_settings)
                self.pc_chan = main.audio.Start(main.audio.LoadSound("assets/sfx/post_combustion.wav"), self.pc_settings)
                self.start = True
                self.pc_started = False
                self.pc_stopped = True
                if self.wind_chan > 0:
                        main.audio.Stop(self.wind_chan)

        def stop_engine(self, main):
                self.turbine_settings.volume = 0
                self.turbine_settings.pitch = 1
                self.air_settings.volume = 0
                self.pc_settings.volume = 0
                main.audio.Stop(self.turbine_chan)
                main.audio.Stop(self.air_chan)
                main.audio.Stop(self.pc_chan)

                self.start = False
                self.pc_started = False
                self.pc_stopped = True

                self.wind_settings.volume = 0
                self.wind_chan = main.audio.Start(main.audio.LoadSound("assets/sfx/wind.wav"), self.wind_settings)

        def update_sfx(self, main, dts):

                level = MathsSupp.get_sound_distance_level(main.camera.GetTransform().GetWorld().GetTranslation(),
                                                                                                   self.aircraft.get_parent_node().GetTransform().GetPosition())

                if self.aircraft.thrust_level > 0 and not self.start:
                        self.start_engine(main)

                if self.aircraft.wreck and not self.exploded:
                        self.explosion_settings.volume = level
                        main.audio.Start(main.audio.LoadSound("assets/sfx/explosion.wav"), self.explosion_settings)
                        self.exploded = True

                if self.start:
                        if self.aircraft.thrust_level <= 0 and False:
                                pass
                        else:
                                self.turbine_settings.volume = 0.5 * level
                                self.turbine_settings.pitch = self.turbine_pitch_levels.x + self.aircraft.thrust_level * (
                                                self.turbine_pitch_levels.y - self.turbine_pitch_levels.x)
                                self.air_settings.volume = (0.1 + self.aircraft.thrust_level * 0.9) * level

                                main.audio.SetChannelState(self.turbine_chan, self.turbine_settings)
                                main.audio.SetChannelState(self.air_chan, self.air_settings)

                                if self.aircraft.post_combution:
                                        self.pc_settings.volume = level
                                        if not self.pc_started:
                                                self.pc_stopped = False
                                                self.pc_settings.volume *= self.pc_cptr / self.pc_start_delay
                                                self.pc_cptr += dts
                                                if self.pc_cptr >= self.pc_start_delay:
                                                        self.pc_started = True
                                                        self.pc_cptr = 0
                                        main.audio.SetChannelState(self.pc_chan, self.pc_settings)
                                else:
                                        if not self.pc_stopped:
                                                self.pc_started = False
                                                self.pc_settings.volume = (1 - self.pc_cptr / self.pc_stop_delay) * level
                                                main.audio.SetChannelState(self.pc_chan, self.pc_settings)
                                                self.pc_cptr += dts
                                                if self.pc_cptr >= self.pc_stop_delay:
                                                        self.pc_stopped = True
                                                        self.pc_cptr = 0
                else:
                        f = min(1, self.aircraft.get_linear_speed() * 3.6 / 1000)
                        self.wind_settings.volume = f * level
                        main.audio.SetChannelState(self.wind_chan, self.wind_settings)

                # Machine gun
                if self.aircraft.gun_machine.num_new > 0:
                        self.machine_gun_settings.volume = level * 0.5
                        self.wind_chan = main.audio.Start(main.audio.LoadSound("assets/sfx/machine_gun.wav"), self.machine_gun_settings)
class Carrier(Destroyable_Machine):
        def __init__(self, name, nationality, plus, scene):
                self.name = name
                Destroyable_Machine.__init__(self, scene.GetNode("aircraft_carrier"), Destroyable_Machine.TYPE_SHIP,
                                                                         nationality)
                self.activated = True
                self.parent_node.GetTransform().SetPosition(hg.Vector3(0, 0, 0))
                self.parent_node.GetTransform().SetRotation(hg.Vector3(0, 0, 0))
                self.radar = scene.GetNode("aircraft_carrier_radar")
                self.rigid, self.collisions = self.get_collisions(scene)
                self.aircraft_start_point = scene.GetNode("carrier_aircraft_start_point")
                self.aircraft_start_point.RemoveComponent(self.aircraft_start_point.GetObject())

        def hit(self, value):
                pass

        def update_kinetics(self, scene, dts):
                rot = self.parent_node.GetTransform().GetRotation()
                # print(str(rot.x)+" , "+str(rot.y)+" , "+str(rot.z))
                rot = self.radar.GetTransform().GetRotation()
                rot.y += radians(45 * dts)
                self.radar.GetTransform().SetRotation(rot)

        def get_aircraft_start_point(self):
                mat = self.aircraft_start_point.GetTransform().GetWorld()
                return mat.GetTranslation(), mat.GetRotation()

        def get_collisions(self, scene):
                rigid = hg.RigidBody()
                rigid.SetType(hg.RigidBodyKinematic)
                self.parent_node.AddComponent(rigid)
                collisions_nodes = []
                for nd in scene.GetNodes():
                        if nd.GetName().find("carrier_col_shape") >= 0:
                                collisions_nodes.append(nd)
                collisions_boxes = []
                for col_shape in collisions_nodes:
                        colbox = hg.BoxCollision()
                        collisions_boxes.append(colbox)
                        obj = col_shape.GetObject()
                        bounds = obj.GetLocalMinMax()
                        dimensions = bounds.mx - bounds.mn
                        pos = col_shape.GetTransform().GetPosition() + bounds.mn + dimensions * 0.5
                        colbox.SetDimensions(dimensions)
                        colbox.SetMatrix(hg.Matrix4.TranslationMatrix(pos))
                        self.parent_node.AddComponent(colbox)
                        scene.RemoveNode(col_shape)
                return rigid, collisions_boxes
class DebugDisplays:
        courbe = []
        ymin = 0
        ymax = 0
        start_courbe = False

        @classmethod
        def maj_courbe(cls, y):
                if not cls.start_courbe:
                        cls.ymin = y
                        cls.ymax = y
                        cls.start_courbe = True
                else:
                        if y < cls.ymin:
                                ymin = y
                        if y > cls.ymax:
                                cls.ymax = y
                cls.courbe.append(y)

        @classmethod
        def affiche_courbe(cls, plus):
                resolution = hg.Vector2(float(plus.GetScreenWidth()), float(plus.GetScreenHeight()))
                num = len(cls.courbe)
                if num > 10:
                        x_step = resolution.x / (num - 1)
                        x1 = 0
                        x2 = x_step
                        y1 = (cls.courbe[0] - cls.ymin) / (cls.ymax - cls.ymin) * resolution.y
                        for i in range(num - 1):
                                y2 = (cls.courbe[i + 1] - cls.ymin) / (cls.ymax - cls.ymin) * resolution.y
                                plus.Line2D(x1, y1, x2, y2, hg.Color.Yellow, hg.Color.Yellow)
                                x1 = x2
                                x2 += x_step
                                y1 = y2

        @classmethod
        def get_2d(cls, camera, renderer, point3d: hg.Vector3):
                f, pos = hg.Project(camera.GetTransform().GetWorld(), camera.GetCamera().GetZoomFactor(),
                                                        renderer.GetAspectRatio(), point3d)
                if f:
                        return hg.Vector2(pos.x, 1 - pos.y)
                else:
                        return None

        @classmethod
        def affiche_vecteur(cls, plus:hg.Plus, camera, position, direction, unitaire=True, c1=hg.Color.Yellow, c2=hg.Color.Red):
                resolution = hg.Vector2(float(plus.GetScreenWidth()),float(plus.GetScreenHeight()))
                if unitaire:
                        position_b = position + direction.Normalized()
                else:
                        position_b = position + direction
                pA = cls.get_2d(camera, plus.GetRenderer(), position)
                pB = cls.get_2d(camera, plus.GetRenderer(), position_b)
                if pA is not None and pB is not None:
                        plus.Line2D(pA.x * resolution.x, pA.y * resolution.y, pB.x * resolution.x, pB.y * resolution.y, c1, c2)

        @classmethod
        def affiche_repere(cls, plus, camera, position: hg.Vector3, repere: hg.Matrix3):
                cls.affiche_vecteur(plus, camera, position, repere.GetX(), hg.Color.White, hg.Color.Red)
                cls.affiche_vecteur(plus, camera, position, repere.GetY(), hg.Color.White, hg.Color.Green)
                cls.affiche_vecteur(plus, camera, position, repere.GetZ(), hg.Color.White, hg.Color.Blue)
class ViewTrame:
        def __init__(self, distance_min=0, distance_max=1000, tile_size=10, margin=1., focal_margin=1.1):
                self.distance_min = distance_min
                self.distance_max = distance_max
                self.tile_size = tile_size
                self.O = hg.Vector2()
                self.A = hg.Vector2()
                self.B = hg.Vector2()
                self.focal_margin = focal_margin
                self.margin = margin
                self.Oint = hg.IntVector2()
                self.Aint = hg.IntVector2()
                self.Bint = hg.IntVector2()

                self.indiceA, self.indiceB = 0, 0
                self.ymin, self.ymax = 0, 0
                self.dAB, self.dBC, self.dAC = hg.IntVector2(), hg.IntVector2(), hg.IntVector2()
                self.detAB, self.detBC, self.detAC = 0, 0, 0
                self.obs = None
                self.obs2D = None
                self.vertices = []
                self.case = 0
                self.send_position = self.default_send

        def default_send(self, position: hg.Vector2):
                pass

        def update_triangle(self, resolution, position:hg.Vector3, direction:hg.Vector3, zoomFactor):
                self.obs = position
                self.obs2D = hg.Vector2(self.obs.x, self.obs.z)
                dir3D = direction
                dir2D = hg.Vector2(dir3D.x, dir3D.z).Normalized()
                focal_distance = zoomFactor * self.focal_margin
                view_width = 2 * resolution.x / resolution.y  # 2 because screen xmin=-1, screen xmax=1

                distAB = self.distance_max * view_width / focal_distance
                VUab = hg.Vector2(-dir2D.y, dir2D.x)
                dir2D *= self.distance_max
                VUab *= distAB / 2

                self.O = hg.Vector2(self.obs.x, self.obs.z)
                self.A = hg.Vector2(self.obs.x + dir2D.x - VUab.x, self.obs.z + dir2D.y - VUab.y)
                self.B = hg.Vector2(self.obs.x + dir2D.x + VUab.x, self.obs.z + dir2D.y + VUab.y)

                # Margin:
                cx = (self.O.x + self.A.x + self.B.x) / 3
                cy = (self.O.y + self.A.y + self.B.y) / 3
                self.O.x = (self.O.x - cx) * self.margin + cx
                self.O.y = (self.O.y - cy) * self.margin + cy
                self.A.x = (self.A.x - cx) * self.margin + cx
                self.A.y = (self.A.y - cy) * self.margin + cy
                self.B.x = (self.B.x - cx) * self.margin + cx
                self.B.y = (self.B.y - cy) * self.margin + cy

                # Tiled triangle:
                self.Oint = hg.IntVector2(int(round(self.O.x / self.tile_size)), int(round(self.O.y / self.tile_size)))
                self.Aint = hg.IntVector2(int(round(self.A.x / self.tile_size)), int(round(self.A.y / self.tile_size)))
                self.Bint = hg.IntVector2(int(round(self.B.x / self.tile_size)), int(round(self.B.y / self.tile_size)))

                self.vertices = [self.Oint, self.Aint, self.Bint]

                self.indiceA = 0
                self.ymin = self.Oint.y
                self.ymax = self.ymin
                if self.Aint.y < self.ymin:
                        self.ymin = self.Aint.y
                        self.indiceA = 1

                if self.Bint.y < self.ymin:
                        self.ymin = self.Bint.y
                        self.indiceA = 2
                if self.Aint.y > self.ymax:
                        self.ymax = self.Aint.y
                if self.Bint.y > self.ymax:
                        self.ymax = self.Bint.y

                self.indiceB = (self.indiceA + 1) % 3
                self.indiceC = (self.indiceA + 2) % 3

                if self.vertices[self.indiceA].y == self.vertices[self.indiceC].y:
                        self.indiceA = self.indiceC
                        self.indiceB = (self.indiceA + 1) % 3
                        self.indiceC = (self.indiceA + 2) % 3

                self.dAB.x = self.vertices[self.indiceB].x - self.vertices[self.indiceA].x
                self.dAB.y = self.vertices[self.indiceB].y - self.vertices[self.indiceA].y
                self.dBC.x = self.vertices[self.indiceC].x - self.vertices[self.indiceB].x
                self.dBC.y = self.vertices[self.indiceC].y - self.vertices[self.indiceB].y
                self.dAC.x = self.vertices[self.indiceC].x - self.vertices[self.indiceA].x
                self.dAC.y = self.vertices[self.indiceC].y - self.vertices[self.indiceA].y

                if self.dAB.y == 0:
                        self.detAB = 0
                else:
                        self.detAB = float(self.dAB.x) / float(self.dAB.y)

                if self.dBC.y == 0:
                        self.detBC = 0
                else:
                        self.detBC = float(self.dBC.x) / float(self.dBC.y)

                if self.dAC.y == 0:
                        self.detAC = 0
                else:
                        self.detAC = float(self.dAC.x) / float(self.dAC.y)

        def fill_triangle(self):
                # Cas1:
                #       A*******B
                #        *****
                #          C
                if self.dAB.y == 0:
                        self.case = 1
                        self.fill_case(self.ymin, self.ymax, float(self.vertices[self.indiceA].x),
                                                   float(self.vertices[self.indiceB].x), self.detAC, self.detBC)
                # Cas2:
                #       A
                #     *****
                #   C*******B
                elif self.dBC.y == 0:
                        self.case = 2
                        self.fill_case(self.ymin, self.ymax, float(self.vertices[self.indiceA].x),
                                                   float(self.vertices[self.indiceA].x), self.detAC, self.detAB)

                # Cas3:
                #        A
                #       ***
                #      C***
                #       ****
                #         *B
                elif self.dAB.y > self.dAC.y:
                        self.case = 3
                        self.fill_case(self.ymin, self.vertices[self.indiceC].y - 1, float(self.vertices[self.indiceA].x),
                                                   float(self.vertices[self.indiceA].x), self.detAC, self.detAB)
                        self.fill_case(self.vertices[self.indiceC].y, self.ymax, float(self.vertices[self.indiceC].x),
                                                   float(self.vertices[self.indiceA].x) + self.detAB * (
                                                                   float(self.vertices[self.indiceC].y) - float(self.ymin)), self.detBC, self.detAB)
                # Cas4:
                #       A
                #       ***
                #       ***B
                #      ****
                #      C*
                else:
                        self.case = 4
                        self.fill_case(self.ymin, self.vertices[self.indiceB].y - 1, float(self.vertices[self.indiceA].x),
                                                   float(self.vertices[self.indiceA].x), self.detAC, self.detAB)

                        self.fill_case(self.vertices[self.indiceB].y, self.ymax,
                                                   float(self.vertices[self.indiceA].x) + self.detAC * (
                                                                   float(self.vertices[self.indiceB].y) - float(self.ymin)),
                                                   float(self.vertices[self.indiceB].x), self.detAC, self.detBC)

        def fill_case(self, ymin, ymax, p_x0, p_x1, d1, d2):
                x0 = p_x0
                x1 = p_x1
                for y in range(ymin, ymax + 1):
                        for x in range(int(x0), int(x1)):
                                pos = hg.Vector2(x * self.tile_size, y * self.tile_size)
                                if (pos - self.obs2D).Len() >= self.distance_min:
                                        self.send_position(pos)
                        x0 += d1
                        x1 += d2


class CloudsLayer(ViewTrame):
        billboard2D = 0
        billboard3D = 1

        def __getstate__(self):
                dico = {"name": self.name, "billboard_type": self.billboard_type,
                        "particles_scale_range": vec2_to_list(self.particles_scale_range), "num_particles": self.num_particles,
                        "num_geometries": self.num_geometries, "particles_files_names": self.particles_files_names,
                        "distance_min": self.distance_min, "distance_max": self.distance_max, "tile_size": self.tile_size,
                        "margin": self.margin, "focal_margin": self.focal_margin, "absorption": self.absorption,
                        "altitude": self.altitude, "altitude_floor": self.altitude_floor, "alpha_threshold": self.alpha_threshold,
                        "scale_falloff": self.scale_falloff, "alpha_scale_falloff": self.alpha_scale_falloff,
                        "altitude_falloff": self.altitude_falloff, "perturbation": self.perturbation,
                    "particles_rot_speed": self.particles_rot_speed, "morph_level": self.morph_level}
                return dico

        def __setstate__(self, state):
                if "particles_scale_range" in state:
                        state["particles_scale_range"] = list_to_vec2(state["particles_scale_range"])
                for k, v in state.items():
                        if hasattr(self, k): setattr(self, k, v)

        def __init__(self, plus, scene, parameters: dict):
                ViewTrame.__init__(self, parameters["distance_min"], parameters["distance_max"], parameters["tile_size"],
                                                   parameters["margin"], parameters["focal_margin"])
                self.name = parameters["name"]
                self.billboard_type = parameters["billboard_type"]
                self.particles_scale_range = list_to_vec2(parameters["particles_scale_range"])
                self.num_tiles = 0
                self.num_particles = parameters["num_particles"]  # List !
                self.num_geometries = parameters["num_geometries"]
                self.particle_index = [0] * self.num_geometries
                self.particle_index_prec = [0] * self.num_geometries
                self.particles_files_names = parameters["particles_files_names"]  # List !
                self.particles = []
                for i in range(0, self.num_geometries):
                        particles = self.create_particles(plus, scene, self.particles_files_names[i], self.num_particles[i],
                                                                                          self.name + "." + str(i))
                        self.particles.append(particles)

                self.absorption = parameters["absorption"]
                self.altitude = parameters["altitude"]
                self.altitude_floor = parameters["altitude_floor"]
                self.alpha_threshold = parameters["alpha_threshold"]
                self.scale_falloff = parameters["scale_falloff"]
                self.alpha_scale_falloff = parameters["alpha_scale_falloff"]
                self.altitude_falloff = parameters["altitude_falloff"]
                self.perturbation = parameters["perturbation"]
                self.particles_rot_speed=0.1
                if "particles_rot_speed" in parameters:
                        self.particles_rot_speed=parameters["particles_rot_speed"]

                # map:
                self.map_size = None
                self.map_scale = None
                self.bitmap_clouds = None

                # Environment:
                self.sun_dir = None
                self.sun_color = None
                self.ambient_color = None

                # Updates vars
                self.rot_hash = hg.Vector3(313.464, 7103.3, 4135.1)
                self.scale_size = 0
                self.pc = hg.Color(1, 1, 1, 1)
                self.cam_pos = None
                self.t = 0

                self.morph_level=1.2
                if "morph_level" in parameters:
                        self.morph_level=parameters["morph_level"]
                self.offset=hg.Vector2(0,0) #Used for clouds wind displacement

                self.renderable_system = scene.GetRenderableSystem()
                self.smooth_alpha_threshold_step=0.1

        @staticmethod
        def create_particles(plus, scene, file_name, num, name):
                particles = []
                for i in range(num):
                        node, geo = load_object(plus, file_name, name + "." + str(i), True)
                        # scene.AddNode(node)
                        particles.append([geo, hg.Matrix4(),geo.GetMaterial(0)])
                return particles

        def set_map(self, bitmap: hg.Picture, map_scale: hg.Vector2, map_position:hg.Vector2):
                self.bitmap_clouds = bitmap
                self.map_scale = map_scale
                self.map_size = hg.IntVector2(self.bitmap_clouds.GetWidth(), self.bitmap_clouds.GetHeight())
                self.offset=map_position

        def set_environment(self, sun_dir, sun_color, ambient_color):
                self.sun_dir = sun_dir
                self.sun_color = sun_color
                self.ambient_color = ambient_color
                self.update_particles()

        def update_lighting(self, sun_dir, sun_color, ambient_color):
                self.sun_dir = sun_dir
                self.sun_color = sun_color
                self.ambient_color = ambient_color
                self.update_particles_lighting()

        def clear_particles(self):
                return
                for particles in self.particles:
                        for particle in particles:
                                particle.SetEnabled(False)

        def update_particles_lighting(self):
                for i in range(0, self.num_geometries):
                        particles = self.particles[i]
                        for particle in particles:
                                material = particle[0].GetMaterial(0)
                                material.SetFloat3("sun_dir", self.sun_dir.x, self.sun_dir.y, self.sun_dir.z)
                                material.SetFloat3("sun_color", self.sun_color.r, self.sun_color.g, self.sun_color.b)
                                material.SetFloat3("ambient_color", self.ambient_color.r, self.ambient_color.g, self.ambient_color.b)

        def update_particles(self):
                altitude_min = self.altitude - self.particles_scale_range.y / 2
                if altitude_min > self.altitude:
                        altitude_max = altitude_min
                        altitude_min = self.altitude
                else:
                        altitude_max = self.altitude
                for i in range(0, self.num_geometries):
                        particles = self.particles[i]
                        c = hg.Color(1., 1., 1., 1.)
                        for particle in particles:
                                material = particle[0].GetMaterial(0)
                                t = material.IsReadyOrFailed()
                                if t:
                                        material.SetFloat("alpha_cloud", c.a)
                                        material.SetFloat3("sun_dir", self.sun_dir.x, self.sun_dir.y, self.sun_dir.z)
                                        material.SetFloat3("sun_color", self.sun_color.r, self.sun_color.g, self.sun_color.b)
                                        material.SetFloat3("ambient_color", self.ambient_color.r, self.ambient_color.g, self.ambient_color.b)
                                        material.SetFloat("absorption_factor", self.absorption)
                                        material.SetFloat("layer_dist", self.distance_min)
                                        material.SetFloat("altitude_min", altitude_min)
                                        material.SetFloat("altitude_max", altitude_max)
                                        material.SetFloat("altitude_falloff", self.altitude_falloff)
                                        material.SetFloat("rot_speed",self.particles_rot_speed)




        def set_altitude(self, value):
                self.altitude = value
                self.update_particles()

        def set_particles_rot_speed(self,value):
                self.particles_rot_speed = value
                self.update_particles()

        def set_distance_min(self, value):
                self.distance_min = value
                self.distance_max = max(self.distance_max, self.distance_min + self.tile_size)
                self.update_particles()

        def set_distance_max(self, value):
                self.distance_max = value
                self.distance_min = min(self.distance_min, self.distance_max - self.tile_size)
                self.update_particles()

        def set_absorption(self, value):
                self.absorption = value
                self.update_particles()

        def set_altitude_floor(self, value):
                self.altitude_floor = value
                self.update_particles()

        def set_altitude_falloff(self, value):
                self.altitude_falloff = value
                self.update_particles()

        def set_particles_min_scale(self, value):
                self.particles_scale_range.x = value
                self.particles_scale_range.y = max(self.particles_scale_range.y, value + 1)
                self.update_particles()

        def set_particles_max_scale(self, value):
                self.particles_scale_range.y = value
                self.particles_scale_range.x = min(self.particles_scale_range.x, value - 1)
                self.update_particles()

        def get_pixel_bilinear(self, pos: hg.Vector2):
                x = (pos.x * self.map_size.x - 0.5) % self.map_size.x
                y = (pos.y * self.map_size.y - 0.5) % self.map_size.y
                xi = int(x)
                yi = int(y)
                xf = x - xi
                yf = y - yi
                xi1 = (xi + 1) % self.map_size.x
                yi1 = (yi + 1) % self.map_size.y
                c1 = self.bitmap_clouds.GetPixelRGBA(xi, yi)
                c2 = self.bitmap_clouds.GetPixelRGBA(xi1, yi)
                c3 = self.bitmap_clouds.GetPixelRGBA(xi, yi1)
                c4 = self.bitmap_clouds.GetPixelRGBA(xi1, yi1)
                c12 = c1 * (1 - xf) + c2 * xf
                c34 = c3 * (1 - xf) + c4 * xf
                c = c12 * (1 - yf) + c34 * yf
                return c

        def update(self, t, camera, resolution,map_position:hg.Vector2):
                self.offset=map_position
                self.t = t
                self.num_tiles = 0
                self.particle_index_prec = self.particle_index
                self.particle_index = [0] * self.num_geometries
                # for i in range (0,self.num_textures_layer_2):
                #    self.particle_index_layer_2.append(0)
                self.cam_pos = camera.GetTransform().GetPosition()
                self.pc = hg.Color(1., 1., 1., 1.)
                self.scale_size = self.particles_scale_range.y - self.particles_scale_range.x
                self.rot_hash = hg.Vector3(133.464, 4713.3, 1435.1)

                self.send_position = self.set_particle
                self.update_triangle(resolution, camera.GetTransform().GetPosition()+hg.Vector3(self.offset.x,0,self.offset.y),
                                     camera.GetTransform().GetWorld().GetZ(), camera.GetCamera().GetZoomFactor())
                self.fill_triangle()


        def set_particle(self, position: hg.Vector2):
                self.num_tiles += 1
                # x = int(position.x / self.map_scale.x * self.map_size.x)
                # y = int(position.y / self.map_scale.y * self.map_size.y)
                # c = self.bitmap_clouds.GetPixelRGBA(int(x % self.map_size.x), int(y % self.map_size.y))
                c = self.get_pixel_bilinear((position+self.offset*self.morph_level) / self.map_scale)

                scale_factor = pow(c.x, self.scale_falloff)
                # id = int(max(0,scale_factor - self.layer_2_alpha_threshold) / (1 - self.layer_2_alpha_threshold) * 7)
                id = int((sin(position.x * 1735.972 + position.y * 345.145) * 0.5 + 0.5) * (self.num_geometries - 1))
                if self.particle_index[id] < self.num_particles[id]:

                        particle = self.particles[id][self.particle_index[id]]
                        if c.x > self.alpha_threshold:
                                smooth_alpha_threshold=min(1,(c.x-self.alpha_threshold)/self.smooth_alpha_threshold_step)
                                s = self.particles_scale_range.x + scale_factor * self.scale_size
                                py = self.altitude + s * self.altitude_floor + (self.perturbation * (1 - c.x) * sin(position.x * 213))
                                pos = hg.Vector3(position.x-self.offset.x, py, position.y-self.offset.y)

                                d = (hg.Vector2(pos.x, pos.z) - hg.Vector2(self.cam_pos.x, self.cam_pos.z)).Len()
                                layer_n = abs(max(0, min(1, (d - self.distance_min) / (self.distance_max - self.distance_min))) * 2 - 1)
                                self.pc.a = (1 - min(pow(layer_n, 8), 1)) * (1 - pow(1. - scale_factor, self.alpha_scale_falloff)) * smooth_alpha_threshold

                                particle[1] = hg.Matrix4(hg.Matrix3.Identity)
                                particle[1].SetScale(hg.Vector3(s, s, s))

                                particle[1].SetTranslation(pos)
                                material = particle[2]
                                material.SetFloat2("pos0", position.x, position.y)
                                material.SetFloat("alpha_cloud", self.pc.a)

                                self.renderable_system.DrawGeometry(particle[0], particle[1])

                                self.particle_index[id] += 1


class Clouds:
        def __setstate__(self, state):
                vec2_list = ["map_scale", "map_position", "v_wind"]
                for k in vec2_list:
                        if k in state: state[k] = list_to_vec2(state[k])
                if "layers" in state:
                        for layer_state in state["layers"]:
                                layer = self.get_layer_by_name(layer_state["name"])
                                if layer is not None:
                                        layer.__setstate__(layer_state)  # !!! Ne recharge pas les géométries !!
                del state["layers"]

                for k, v in state.items():
                        if hasattr(self, k): setattr(self, k, v)
                self.update_layers_cloud_map()
                self.update_layers_environment()
                self.update_particles()

        def __getstate__(self):
                layers_list = []
                for layer in self.layers:
                        layers_list.append(layer.__getstate__())
                dico = {"name": self.name, "bitmap_clouds_file": self.bitmap_clouds_file,
                        "map_scale": vec2_to_list(self.map_scale), "map_position": vec2_to_list(self.map_position),
                    "v_wind":vec2_to_list(self.v_wind),"layers": layers_list}
                return dico

        def __init__(self, plus, scene, main_light, resolution, parameters):

                self.layers = []
                for layer_params in parameters["layers"]:
                        self.layers.append(CloudsLayer(plus, scene, layer_params))

                self.name = parameters["name"]
                self.t = 0
                self.cam_pos = None
                self.bitmap_clouds = hg.Picture()
                self.bitmap_clouds_file = parameters["bitmap_clouds_file"]
                hg.LoadPicture(self.bitmap_clouds, self.bitmap_clouds_file)
                self.map_size = hg.IntVector2(self.bitmap_clouds.GetWidth(), self.bitmap_clouds.GetHeight())
                self.map_scale = list_to_vec2(parameters["map_scale"])
                self.map_position = list_to_vec2(parameters["map_position"])
                self.sun_light = main_light
                self.ambient_color = scene.GetEnvironment().GetAmbientColor() * scene.GetEnvironment().GetAmbientIntensity()
                self.v_wind=hg.Vector2(60,60)
                if "v_wind" in parameters:
                        self.v_wind=list_to_vec2(parameters["v_wind"])

                self.update_layers_cloud_map()
                self.update_layers_environment()
                self.update_particles()

        def get_layer_by_name(self, layer_name):
                for layer in self.layers:
                        if layer.name == layer_name: return layer
                return None

        def update_layers_cloud_map(self):
                for layer in self.layers:
                        layer.set_map(self.bitmap_clouds, self.map_scale, self.map_position)

        def update_layers_lighting(self):
                sun_dir = self.sun_light.GetTransform().GetWorld().GetZ()
                lt = self.sun_light.GetLight()
                sun_color = lt.GetDiffuseColor() * lt.GetDiffuseIntensity()
                for layer in self.layers:
                        layer.update_lighting(sun_dir, sun_color, self.ambient_color)

        def update_layers_environment(self):
                sun_dir = self.sun_light.GetTransform().GetWorld().GetZ()
                lt = self.sun_light.GetLight()
                sun_color = lt.GetDiffuseColor() * lt.GetDiffuseIntensity()
                for layer in self.layers:
                        layer.set_environment(sun_dir, sun_color, self.ambient_color)

        def load_json_script(self, file_name="assets/scripts/clouds_parameters.json"):
                json_script = hg.GetFilesystem().FileToString(file_name)
                if json_script != "":
                        script_parameters = json.loads(json_script)
                        self.__setstate__(script_parameters)

        def save_json_script(self, scene, output_filename="assets/scripts/clouds_parameters.json"):
                script_parameters = self.__getstate__()
                json_script = json.dumps(script_parameters, indent=4)
                return hg.GetFilesystem().StringToFile(output_filename, json_script)

        def set_map_scale_x(self, value):
                self.map_scale.x = value
                self.update_layers_cloud_map()

        def set_map_scale_z(self, value):
                self.map_scale.y = value
                self.update_layers_cloud_map()

        def clear_particles(self):
                for layer in self.layers:
                        layer.clear_particles()

        def update_particles(self):
                for layer in self.layers:
                        layer.update_particles()

        def update(self, t, delta_t, scene, resolution):
                self.t = t
                camera = scene.GetCurrentCamera()
                self.cam_pos = camera.GetTransform().GetPosition()
                self.map_position+=self.v_wind*delta_t
                for layer in self.layers:
                        layer.update(t, camera, resolution, self.map_position)
track_position = hg.Vector3(0, 4, -20)
track_orientation = hg.Matrix3(hg.Vector3(1, 0, 0), hg.Vector3(0, 1, 0), hg.Vector3(0, 0, 1))

pos_inertia = 0.2
rot_inertia = 0.07

follow_inertia = 0.01
follow_distance = 200

target_point = hg.Vector3(0, 0, 0)
target_matrix = hg.Matrix3()
target_node = None

satellite_camera = None
satellite_view_size = 100
satellite_view_size_inertia = 0.7

camera_move = hg.Vector3(0, 0, 0)  # Translation in frame

noise_x = Temporal_Perlin_Noise(0.1446)
noise_y = Temporal_Perlin_Noise(0.1017)
noise_z = Temporal_Perlin_Noise(0.250314)
back_view = {"position": hg.Vector3(0, 4, -20),
                         "orientation": hg.Matrix3(hg.Vector3(1, 0, 0), hg.Vector3(0, 1, 0), hg.Vector3(0, 0, 1)),
                         "pos_inertia": 0.2, "rot_inertia": 0.07}

front_view = {"position": hg.Vector3(0, 4, 40),
                          "orientation": hg.Matrix3(hg.Vector3(-1, 0, 0), hg.Vector3(0, 1, 0), hg.Vector3(0, 0, -1)),
                          "pos_inertia": 0.9, "rot_inertia": 0.05}

right_view = {"position": hg.Vector3(-40, 4, 0),
                          "orientation": hg.Matrix3(hg.Vector3(0, 0, -1), hg.Vector3(0, 1, 0), hg.Vector3(1, 0, 0)),
                          "pos_inertia": 0.9, "rot_inertia": 0.05}

left_view = {"position": hg.Vector3(40, 4, 0),
                         "orientation": hg.Matrix3(hg.Vector3(0, 0, 1), hg.Vector3(0, 1, 0), hg.Vector3(-1, 0, 0)),
                         "pos_inertia": 0.9, "rot_inertia": 0.05}

top_view = {"position": hg.Vector3(0, 50, 0),
                        "orientation": hg.Matrix3(hg.Vector3(1, 0, 0), hg.Vector3(0, 0, 1), hg.Vector3(0, -1, 0)),
                        "pos_inertia": 0.9, "rot_inertia": 0.05}
res_w=520
res_h=160
monitors=None
monitors_names=[]
modes=None
current_monitor=0
current_mode=0
ratio_filter=0
flag_windowed=False

screenModes=[hg.FullscreenMonitor1,hg.FullscreenMonitor2,hg.FullscreenMonitor3]
smr_screenMode=hg.FullscreenMonitor1
smr_resolution=hg.IntVector2(1280,1024)
def list_to_color(c: list):
        return hg.Color(c[0], c[1], c[2], c[3])

def color_to_list(c: hg.Color):
        return [c.r, c.g, c.b, c.a]

def list_to_vec2(v: list):
        return hg.Vector2(v[0], v[1])

def vec2_to_list(v: hg.Vector2):
        return [v.x, v.y]

def list_to_vec3(v: list):
        return hg.Vector3(v[0], v[1],v[2])

def list_to_vec3_radians(v: list):
        v=list_to_vec3(v)
        v.x=radians(v.x)
        v.y=radians(v.y)
        v.z=radians(v.z)
        return v

def vec3_to_list(v: hg.Vector3):
        return [v.x, v.y, v.z]

def vec3_to_list_degrees(v: hg.Vector3):
        l=vec3_to_list(v)
        l[0]=degrees(l[0])
        l[1]=degrees(l[1])
        l[2]=degrees(l[2])
        return l

def load_json_matrix(file_name):
        json_script = hg.GetFilesystem().FileToString(file_name)
        if json_script != "":
                script_parameters = json.loads(json_script)
                pos = list_to_vec3(script_parameters["position"])
                rot = list_to_vec3_radians(script_parameters["rotation"])
                return pos,rot
        return None,None

def save_json_matrix(pos : hg.Vector3, rot:hg.Vector3,output_filename ):
        script_parameters = {"position" : vec3_to_list(pos), "rotation" : vec3_to_list_degrees(rot)}
        json_script = json.dumps(script_parameters, indent=4)
        return hg.GetFilesystem().StringToFile(output_filename, json_script)

def duplicate_node_object(original_node:hg.Node, name):
        node = hg.Node(name)
        trans = hg.Transform()
        node.AddComponent(trans)
        obj = hg.Object()
        obj.SetGeometry(original_node.GetObject().GetGeometry())
        node.AddComponent(obj)
        return node

def load_object(plus,geometry_file_name, name,duplicate_material=False):
        renderSystem = plus.GetRenderSystem()
        node = hg.Node(name)
        trans = hg.Transform()
        node.AddComponent(trans)
        obj = hg.Object()
        geo = hg.Geometry()
        hg.LoadGeometry(geo,geometry_file_name)
        if geo is not None:
                geo = renderSystem.CreateGeometry(geo,False)
                if duplicate_material:
                        material = geo.GetMaterial(0)
                        material = material.Clone()
                        geo.SetMaterial(0,material)
                obj.SetGeometry(geo)
                node.AddComponent(obj)
        return node,geo
def gui_ScreenModeRequester():
        global flag_windowed
        global current_monitor,current_mode,monitors_names,modes

        hg.ImGuiSetNextWindowPosCenter(hg.ImGuiCond_Always)
        hg.ImGuiSetNextWindowSize(hg.Vector2(res_w, res_h), hg.ImGuiCond_Always)
        if hg.ImGuiBegin("Choose screen mode",hg.ImGuiWindowFlags_NoTitleBar
                                                                                  | hg.ImGuiWindowFlags_MenuBar
                                                                                  | hg.ImGuiWindowFlags_NoMove
                                                                                  | hg.ImGuiWindowFlags_NoSavedSettings
                                                                                  | hg.ImGuiWindowFlags_NoCollapse):
                if hg.ImGuiBeginCombo("Monitor", monitors_names[current_monitor]):
                        for i in range(len(monitors_names)):
                                f = hg.ImGuiSelectable(monitors_names[i], current_monitor == i)
                                if f:
                                        current_monitor = i
                        hg.ImGuiEndCombo()

                if hg.ImGuiBeginCombo("Screen size", modes[current_monitor][current_mode].name):
                        for i in range(len(modes[current_monitor])):
                                f = hg.ImGuiSelectable(modes[current_monitor][i].name+"##"+str(i), current_mode == i)
                                if f:
                                        current_mode = i
                        hg.ImGuiEndCombo()
                ok=hg.ImGuiButton("Ok")
                hg.ImGuiSameLine()
                cancel=hg.ImGuiButton("Quit")

        hg.ImGuiEnd()

        if ok: return "ok"
        elif cancel: return "quit"
        else: return ""
def request_screen_mode(p_ratio_filter=0):
        global monitors,monitors_names,modes,smr_screenMode,smr_resolution,ratio_filter

        ratio_filter=p_ratio_filter
        monitors = hg.GetMonitors()
        monitors_names = []
        modes = []
        for i in range(monitors.size()):
                monitors_names.append(hg.GetMonitorName(monitors.at(i))+str(i))
                f, m = hg.GetMonitorModes(monitors.at(i))
                filtered_modes=[]
                for j in range(m.size()):
                        md=m.at(j)
                        rect = md.rect
                        epsilon = 0.01
                        r = (rect.ex - rect.sx) / (rect.ey - rect.sy)
                        if ratio_filter == 0 or r - epsilon < ratio_filter < r + epsilon:
                                filtered_modes.append(md)
                modes.append(filtered_modes)

        plus=hg.GetPlus()
        plus.RenderInit(res_w, res_h, 1, hg.Windowed)
        select=""
        while select=="":
                select=gui_ScreenModeRequester()
                plus.Flip()
                plus.EndFrame()
        plus.RenderUninit()

        if select=="ok":
                smr_screenMode=hg.Windowed
                rect=modes[current_monitor][current_mode].rect
                smr_resolution.x,smr_resolution.y=rect.ex-rect.sx,rect.ey-rect.sy
        return select,smr_screenMode,smr_resolution
def get_2d(camera, renderer, point3d: hg.Vector3):
        f, pos = hg.Project(camera.GetTransform().GetWorld(), camera.GetCamera().GetZoomFactor(), renderer.GetAspectRatio(),
                                                point3d)
        if f:
                return hg.Vector2(pos.x, 1 - pos.y)
        else:
                return None


def update_radar(Main,plus,aircraft,targets):
        value = Main.HSL_postProcess.GetL()
        t=hg.time_to_sec_f(plus.GetClock())
        rx,ry = 150/1600*Main.resolution.x, 150/900*Main.resolution.y
        rs=200/1600*Main.resolution.x
        rm=6/1600*Main.resolution.x

        radar_scale = 4000
        plot_size = 12/1600*Main.resolution.x/2

        plus.Sprite2D(rx,ry,rs,"assets/sprites/radar.png",hg.Color(1,1,1,value))
        mat=aircraft.get_parent_node().GetTransform().GetWorld()

        aZ=mat.GetZ()
        aZ.y=0
        aZ.Normalize()
        aY=mat.GetY()
        if aY.y<0:
                aY=hg.Vector3(0,-1,0)
        else:
                aY=hg.Vector3(0,1,0)
        aX=hg.Cross(aY,aZ)
        mat_trans = hg.Matrix4.TransformationMatrix(mat.GetTranslation(),hg.Matrix3(aX,aY,aZ)).InversedFast()

        for target in targets:
                if not target.wreck and target.activated:
                        t_mat = target.get_parent_node().GetTransform().GetWorld()
                        aZ=t_mat.GetZ()
                        aZ.y=0
                        aZ.Normalize()
                        aY=hg.Vector3(0,1,0)
                        aX=hg.Cross(aY,aZ)
                        t_mat_trans = mat_trans * hg.Matrix4.TransformationMatrix(t_mat.GetTranslation(),hg.Matrix3(aX,aY,aZ))
                        pos=t_mat_trans.GetTranslation()
                        v2D = hg.Vector2(pos.x, pos.z) / radar_scale * rs / 2
                        if abs(v2D.x)<rs/2-rm and abs(v2D.y)<rs/2-rm:

                                if target.type==Destroyable_Machine.TYPE_MISSILE:
                                        plot=Main.texture_hud_plot_missile
                                elif target.type==Destroyable_Machine.TYPE_AIRCRAFT:
                                        plot=Main.texture_hud_plot_aircraft
                                elif target.type==Destroyable_Machine.TYPE_SHIP:
                                        plot=Main.texture_hud_plot_ship
                                t_mat_rot = t_mat_trans.GetRotationMatrix()
                                ps=plot_size
                                a = 0.5 + 0.5*abs(sin(t*uniform(1,500)))
                        else:
                                if target.type == Destroyable_Machine.TYPE_MISSILE: continue
                                dir=v2D.Normalized()
                                v2D = dir * (rs/2-rm)
                                ps=plot_size
                                plot=Main.texture_hud_plot_dir
                                aZ=hg.Vector3(dir.x,0,dir.y)
                                aX=hg.Cross(hg.Vector3.Up,aZ)
                                t_mat_rot = hg.Matrix3(aX,hg.Vector3.Up,aZ)
                                a = 0.5 + 0.5*abs(sin(t*uniform(1,500)))

                        cx, cy = rx + v2D.x, ry + v2D.y

                        if target.nationality == 1:
                                c = hg.Color(0.25, 1., 0.25, a)
                        elif target.nationality == 2:
                                c = hg.Color(1., 0.5, 0.5, a)

                        c.a*=value
                        p1 = t_mat_rot * hg.Vector3(-plot_size, 0, -ps)
                        p2 = t_mat_rot * hg.Vector3(-plot_size, 0, ps)
                        p3 = t_mat_rot * hg.Vector3(plot_size, 0, ps)
                        p4 = t_mat_rot * hg.Vector3(plot_size, 0, -ps)
                        plus.Quad2D(cx + p1.x, cy + p1.z, cx + p2.x, cy + p2.z, cx + p3.x, cy + p3.z, cx + p4.x, cy + p4.z, c, c, c, c, plot)


        c=hg.Color(1,1,1,value*max(pow(1-aircraft.health_level,2),0.05))
        plus.Quad2D(rx-rs/2,ry-rs/2,
                                rx-rs/2,ry+rs/2,
                                rx+rs/2,ry+rs/2,
                                rx+rs/2,ry-rs/2,
                                c,c,c,c,Main.texture_noise,0.25+0.25*sin(t*103),(0.65+0.35*sin(t*83)),0.75+0.25*sin(t*538),0.75+0.25*cos(t*120))
        c=hg.Color(1,1,1,value)
        plus.Sprite2D(rx, ry, rs, "assets/sprites/radar_light.png",hg.Color(1,1,1,0.3*value))
        plus.Sprite2D(rx, ry, rs, "assets/sprites/radar_box.png",c)


def update_machine_gun_sight(Main,plus,aircraft:Aircraft):
        mat = aircraft.get_parent_node().GetTransform().GetWorld()
        aZ=mat.GetZ()
        pos=aircraft.gun_position * mat
        p=pos+aZ*500
        Main.gun_sight_2D=get_2d(Main.scene.GetCurrentCamera(),plus.GetRenderer(),p)
        p2D=Main.gun_sight_2D
        if p2D is not None:
                plus.Sprite2D(p2D.x*Main.resolution.x, p2D.y*Main.resolution.y, 64/1600*Main.resolution.x, "assets/sprites/machine_gun_sight.png", hg.Color(0.5,1,0.5, Main.HSL_postProcess.GetL()))


def update_target_sight(Main,plus,aircraft:Aircraft):
        tps = hg.time_to_sec_f(plus.GetClock())
        target=aircraft.get_target()
        f = Main.HSL_postProcess.GetL()
        if target is not None:
                p2D=get_2d(Main.scene.GetCurrentCamera(),plus.GetRenderer(),target.get_parent_node().GetTransform().GetPosition())
                if p2D is not None:
                        a_pulse = 0.5 if (sin(tps * 20) > 0) else 0.75
                        if aircraft.target_locked:
                                c=hg.Color(1.,0.5,0.5,a_pulse)
                                msg="LOCKED - "+str(int(aircraft.target_distance))
                                x=(p2D.x - 32 / 1600)
                                a=a_pulse
                        else:
                                msg=str(int(aircraft.target_distance))
                                x=(p2D.x - 12 / 1600)
                                c=hg.Color(0.5,1,0.5,0.75)

                        c.a*=f
                        plus.Sprite2D(p2D.x * Main.resolution.x, p2D.y * Main.resolution.y, 32 / 1600 * Main.resolution.x,
                                                  "assets/sprites/target_sight.png", c)


                        if aircraft.target_out_of_range:

                                plus.Text2D((p2D.x-40/1600) * Main.resolution.x, (p2D.y-24/900) * Main.resolution.y, "OUT OF RANGE",
                                                        0.012 * Main.resolution.y, hg.Color(0.2, 1, 0.2, a_pulse*f))
                        else:
                                plus.Text2D(x * Main.resolution.x, (p2D.y - 24 / 900) * Main.resolution.y,
                                                        msg, 0.012 * Main.resolution.y, c)

                        if aircraft.target_locking_state>0:
                                t=sin(aircraft.target_locking_state*pi-pi/2)*0.5+0.5
                                p2D=hg.Vector2(0.5,0.5)*(1-t)+p2D*t
                                plus.Sprite2D(p2D.x * Main.resolution.x, p2D.y * Main.resolution.y, 32 / 1600 * Main.resolution.x,
                                                          "assets/sprites/missile_sight.png", c)

                c=hg.Color(0,1,0,f)
                plus.Text2D(0.05 * Main.resolution.x, 0.93 * Main.resolution.y, "Target dist: %d" % (aircraft.target_distance),
                                        0.016 * Main.resolution.y,c)
                plus.Text2D(0.05 * Main.resolution.x, 0.91 * Main.resolution.y, "Target cap: %d" % (aircraft.target_cap),
                                        0.016 * Main.resolution.y, c)
                plus.Text2D(0.05 * Main.resolution.x, 0.89 * Main.resolution.y, "Target alt: %d" % (aircraft.target_altitude),
                                        0.016 * Main.resolution.y, c)



def display_hud(Main, plus, aircraft: Aircraft,targets):
        f =Main.HSL_postProcess.GetL()
        tps = hg.time_to_sec_f(plus.GetClock())
        a_pulse = 0.1 if (sin(tps * 25) > 0) else 0.9
        hs, vs = aircraft.get_world_speed()

        plus.Text2D(0.05 * Main.resolution.x, 0.95 * Main.resolution.y, "Health: %d" % (aircraft.health_level*127),
                                0.016 * Main.resolution.y, (hg.Color.White*aircraft.health_level+hg.Color.Red*(1-aircraft.health_level)) * f)

        plus.Text2D(0.49 * Main.resolution.x, 0.95 * Main.resolution.y, "Cap: %d" % (aircraft.cap),
                                0.016 * Main.resolution.y, hg.Color.White * f)

        plus.Text2D(0.8 * Main.resolution.x, 0.90 * Main.resolution.y, "Altitude (ft): %d" % (aircraft.get_altitude()*3.28084),
                                0.016 * Main.resolution.y, hg.Color.White * f)
        plus.Text2D(0.8 * Main.resolution.x, 0.88 * Main.resolution.y, "Vertical speed (m/s): %d" % (vs), 0.016 * Main.resolution.y,
                                hg.Color.White * f)

        plus.Text2D(0.8 * Main.resolution.x, 0.03 * Main.resolution.y, "horizontal speed (m/s): %d" % (hs), 0.016 * Main.resolution.y,
                                hg.Color.White * f)

        plus.Text2D(0.8 * Main.resolution.x, 0.13 * Main.resolution.y, "Pitch: %d" % (aircraft.pitch_attitude),
                                0.016 * Main.resolution.y, hg.Color.White * f)

        plus.Text2D(0.8 * Main.resolution.x, 0.11 * Main.resolution.y,
                                "Roll: %d" % (aircraft.roll_attitude), 0.016 * Main.resolution.y,
                                hg.Color.White * f)

        ls=aircraft.get_linear_speed()*3.6
        plus.Text2D(0.8 * Main.resolution.x, 0.05 * Main.resolution.y,
                                "Speed (mph): %d" % (ls*0.621371), 0.016 *Main. resolution.y,
                                hg.Color.White * f)
        if ls<250 and not aircraft.flag_landed:
                plus.Text2D(749/1600 * Main.resolution.x, 120/900 * Main.resolution.y, "LOW SPEED",
                                        0.018 * Main.resolution.y, hg.Color(1.,0,0,a_pulse))

        plus.Text2D(0.8 * Main.resolution.x, 0.01 * Main.resolution.y,
                                "Linear acceleration (m.s2): %.2f" % (Main.p1_aircraft.get_linear_acceleration()), 0.016 * Main.resolution.y,
                                hg.Color.White * f)

        plus.Text2D(749 / 1600 * Main.resolution.x, 94 / 900 * Main.resolution.y,
                                "Throttle: %d" % (Main.p1_aircraft.thrust_level * 127), 0.016 * Main.resolution.y, hg.Color.White * f)
        if Main.p1_aircraft.brake_level > 0:
                plus.Text2D(688 / 1600 * Main.resolution.x, 32 / 900 * Main.resolution.y,
                                        "Brake: %d" % (Main.p1_aircraft.brake_level * 127), 0.016 * Main.resolution.y, hg.Color(1, 0.4, 0.4) * f)
        if Main.p1_aircraft.flaps_level > 0:
                plus.Text2D(824 / 1600 * Main.resolution.x, 32 / 900 * Main.resolution.y,
                                        "Flaps: %d" % (Main.p1_aircraft.flaps_level * 127), 0.016 * Main.resolution.y, hg.Color(1, 0.8, 0.4) * f)


        #if Main.p1_aircraft.post_combution:
        #    plus.Text2D(710 / 1600 * Main.resolution.x, 76 / 900 * Main.resolution.y, "POST COMBUSTION", 0.02 * Main.resolution.y,
        #                hg.Color.Red)

        update_radar(Main, plus, aircraft, targets)
        update_target_sight(Main, plus, aircraft)

        if not Main.satellite_view:
                update_machine_gun_sight(Main, plus, aircraft)

def setup_camera_follow(targetNode: hg.Node, targetPoint: hg.Vector3, targetMatrix: hg.Matrix3):
        global target_point, target_matrix, target_node
        target_point = targetPoint
        target_matrix = targetMatrix
        target_node = targetNode


def RangeAdjust(value, oldmin, oldmax, newmin, newmax):
        return (value - oldmin) / (oldmax - oldmin) * (newmax - newmin) + newmin


def update_target_point(dts):
        global target_point, target_matrix
        v = target_node.GetTransform().GetPosition() - target_point
        target_point += v * pos_inertia * dts * 60

        mat_n = target_node.GetTransform().GetWorld()
        rz = hg.Cross(target_matrix.GetZ(), mat_n.GetZ())
        ry = hg.Cross(target_matrix.GetY(), mat_n.GetY())
        mr = rz + ry
        if mr.Len() > 0.001:
                target_matrix = MathsSupp.rotate_matrix(target_matrix, mr.Normalized(), mr.Len() * rot_inertia * dts * 60)


def update_track_translation(camera: hg.Node, dts):
        global camera_move
        trans = camera.GetTransform()
        camera_pos = trans.GetPosition()
        new_position = target_point + target_matrix.GetX() * track_position.x + target_matrix.GetY() * track_position.y + target_matrix.GetZ() * track_position.z
        trans.SetPosition(new_position)
        camera_move = new_position - camera_pos
        return new_position


def update_follow_translation(camera: hg.Node, dts):
        global camera_move
        trans = camera.GetTransform()
        camera_pos = trans.GetPosition()
        aX = trans.GetWorld().GetX()
        target_pos = target_node.GetTransform().GetPosition()

        # Wall
        v = target_pos - camera_pos
        target_dir = v.Normalized()
        target_dist = v.Len()

        v_trans = target_dir * (target_dist - follow_distance) + aX * 20

        new_position = camera_pos + v_trans * follow_inertia * 60 * dts
        trans.SetPosition(new_position)
        camera_move = new_position - camera_pos
        return new_position


def update_track_direction(camera: hg.Node, dts, noise_level):
        # v = target_point - camera.GetTransform().GetPosition()
        f = radians(noise_level)
        rot = target_matrix.ToEuler()
        rot += hg.Vector3(noise_x.temporal_Perlin_noise(dts) * f, noise_y.temporal_Perlin_noise(dts) * f,
                                          noise_z.temporal_Perlin_noise(dts) * f)
        rot_mat = hg.Matrix3.FromEuler(rot)
        rot_mat = rot_mat * track_orientation
        camera.GetTransform().SetRotationMatrix(rot_mat)
        return rot_mat  # camera.GetTransform().GetWorld().GetRotationMatrix().LookAt(v, target_matrix.GetY()))


def update_follow_direction(camera: hg.Node):
        v = target_point - camera.GetTransform().GetPosition()
        rot_mat = camera.GetTransform().GetWorld().GetRotationMatrix().LookAt(v, hg.Vector3.Up)
        camera.GetTransform().SetRotationMatrix(rot_mat)
        return rot_mat


def update_camera_tracking(camera: hg.Node, dts, noise_level=0):
        global target_point, target_node
        update_target_point(dts)
        rot_mat = update_track_direction(camera, dts, noise_level)
        pos = update_track_translation(camera, dts)
        mat = hg.Matrix4(rot_mat)
        mat.SetTranslation(pos)
        return mat


def update_camera_follow(camera: hg.Node, dts):
        global target_point, target_node
        update_target_point(dts)
        rot_mat = update_follow_direction(camera)
        pos = update_follow_translation(camera, dts)
        mat = hg.Matrix4(rot_mat)
        mat.SetTranslation(pos)
        return mat


def set_track_view(parameters: dict):
        global track_position, track_orientation, pos_inertia, rot_inertia
        track_position = parameters["position"]
        track_orientation = parameters["orientation"]
        pos_inertia = parameters["pos_inertia"]
        rot_inertia = parameters["rot_inertia"]


def setup_satellite_camera(camera: hg.Node):
        camera.GetCamera().SetOrthographic(True)
        camera.GetCamera().SetOrthographicSize(satellite_view_size)
        camera.GetTransform().SetRotation(hg.Vector3(radians(90), 0, 0))


def update_satellite_camera(camera, screen_ratio, dts):
        camera.GetTransform().SetPosition(
                hg.Vector3(target_point.x, camera.GetCamera().GetOrthographicSize() * screen_ratio, target_point.z))
        cam = camera.GetCamera()
        size = cam.GetOrthographicSize()
        cam.SetOrthographicSize(size + (satellite_view_size - size) * satellite_view_size_inertia)


def increment_satellite_view_size():
        global satellite_view_size
        satellite_view_size *= 1.01


def decrement_satellite_view_size():
        global satellite_view_size
        satellite_view_size = max(10, satellite_view_size / 1.01)

class Main:
        resolution = hg.Vector2(1600, 900)
        antialiasing = 2
        screenMode = hg.FullscreenMonitor1
        
        main_node = hg.Node()

        controller = None

        scene = None
        camera = None
        satellite_camera = None
        camera_matrix = None
        camera_v_move = hg.Vector3(0, 0, 0)  # Camera velocity for sfx
        fps = None
        sea_render = None
        ligth_sun = None
        ligth_sky = None

        sea_render_script = None
        clouds_render_script = None

        water_reflection = None

        p1_aircraft = None
        p2_aircraft = None

        p1_success = False

        carrier = None
        carrier_radar = None
        island = None

        p1_missiles = [None] * 4
        p2_missiles = [None] * 4
        p1_missiles_smoke_color = hg.Color(1, 1, 1, 1)
        p2_missiles_smoke_color = hg.Color(1, 1, 1, 1)

        p1_targets = []

        bullets = None
        ennemy_bullets = None

        title_font = "assets/fonts/destroy.ttf"
        hud_font = "assets/fonts/Furore.otf"
        texture_hud_plot = None
        texture_noise = None

        fading_cptr = 0
        fading_start_saturation = 0
        fadout_flag = False
        fadout_cptr = 0

        audio = None
        p1_sfx = None
        p2_sfx = None

        title_music = 0
        title_music_settings = None

        clouds = None
        render_volumetric_clouds = True

        show_debug_displays = False
        display_gui = False

        satellite_view = False

        HSL_postProcess=None
        MotionBlur_postProcess=None
        RadialBlur_postProcess=None

        flag_MotionBlur=False

        radial_blur_strength = 0.5
        deceleration_blur_strength=1/6
        acceleration_blur_strength=1/3

        gun_sight_2D=None

        current_view=None
def init_game(plus):
        init_scene(plus)
        Aircraft.main_node = Main.main_node

        Main.audio = hg.CreateMixer()
        Main.audio.Open()

        # Clear color alpha = 0
        Main.scene.GetEnvironment().SetBackgroundColor(hg.Color(0, 0, 0, 0))
        # Aircrafts & Cie:
        Main.p1_aircraft = Aircraft("TangoCharly", 1, "aircraft", plus, Main.scene, hg.Vector3(0, 3000, 0),
                                                                hg.Vector3(0, 0, 0))
        Main.p2_aircraft = Aircraft("Zorglub", 2, "ennemyaircraft", plus, Main.scene, hg.Vector3(4000, 3000, 4000),
                                                                hg.Vector3(0, 0, 0))
        Main.carrier = Carrier("Charles_de_Gaules", 1, plus, Main.scene)

        for i in range(4):
                Main.p1_missiles[i] = Missile("sidewinder_" + str(i), 1, plus, Main.scene, Main.audio,
                                                                          "assets/weaponry/enemymissile_low.geo", "assets/weaponry/enemymissile_smoke")
                Main.p2_missiles[i] = Missile("ennemy_sidewinder_" + str(i), 2, plus, Main.scene, Main.audio,
                                                                          "assets/weaponry/enemymissile_low.geo", "assets/weaponry/enemymissile_smoke")

        # Machine guns :
        Main.bullets = Main.p1_aircraft.gun_machine
        Main.ennemy_bullets = Main.p2_aircraft.gun_machine

        Main.p1_aircraft.set_destroyable_targets([Main.p2_aircraft, Main.carrier])
        Main.p2_aircraft.set_destroyable_targets([Main.p1_aircraft, Main.carrier])

        # Fps
        Main.fps = hg.FPSController(0, 0, 0)

        Main.controller=find_controller("xinput.port0")
        Main.scene.Commit()
        Main.scene.WaitCommit()
        #plus.UpdateScene(Main.scene)

        load_game_parameters()

        Main.texture_hud_plot_aircraft = plus.LoadTexture("assets/sprites/plot.png")
        Main.texture_hud_plot_missile = plus.LoadTexture("assets/sprites/plot_missile.png")
        Main.texture_hud_plot_ship = plus.LoadTexture("assets/sprites/plot_ship.png")
        Main.texture_hud_plot_dir = plus.LoadTexture("assets/sprites/plot_dir.png")
        Main.texture_noise = plus.LoadTexture("assets/sprites/noise.png")

        # ---- Blend settings:

        renderer = plus.GetRenderer()
        renderer.SetBlendFunc(hg.BlendSrcAlpha, hg.BlendOneMinusSrcAlpha)

        # --- Sfx:
        Main.p1_sfx = AircraftSFX(Main.p1_aircraft)
        Main.p2_sfx = AircraftSFX(Main.p2_aircraft)
        # P2 engine sound is different:
        Main.p2_sfx.set_air_pitch(0.75)
        Main.p2_sfx.set_pc_pitch(1.5)
        Main.p2_sfx.set_turbine_pitch_levels(hg.Vector2(1.5, 2.5))

        # ---- Camera:
        Main.scene.SetCurrentCamera(Main.camera)

        # ---- PostProcess:
        Main.MotionBlur_postProcess=hg.MotionBlurPostProcess()
        Main.HSL_postProcess=hg.HSLPostProcess()
        Main.RadialBlur_postProcess=hg.RadialBlurPostProcess()

        Main.camera.AddComponent(Main.RadialBlur_postProcess)
        Main.camera.AddComponent(Main.HSL_postProcess)

        post_processes_load_parameters()


def set_p1_missiles_smoke_color(color: hg.Color):
        Main.p1_missiles_smoke_color = color
        for missile in Main.p1_missiles:
                missile.set_smoke_solor(color)


def set_p2_missiles_smoke_color(color: hg.Color):
        Main.p2_missiles_smoke_color = color
        for missile in Main.p2_missiles:
                missile.set_smoke_solor(color)


def set_p1_gun_color(color: hg.Color):
        Main.bullets.colors = [color]


def set_p2_gun_color(color: hg.Color):
        Main.ennemy_bullets.colors = [color]
def load_scene_parameters(file_name="assets/scripts/scene_parameters.json"):
        json_script = hg.GetFilesystem().FileToString(file_name)
        environment = Main.scene.GetEnvironment()
        if json_script != "":
                script_parameters = json.loads(json_script)
                Main.ligth_sun.GetLight().SetDiffuseColor(list_to_color(script_parameters["sunlight_color"]))
                Main.ligth_sky.GetLight().SetDiffuseColor(list_to_color(script_parameters["skylight_color"]))
                environment.SetAmbientColor(list_to_color(script_parameters["ambient_color"]))
                environment.SetAmbientIntensity(script_parameters["ambient_intensity"])
                Main.render_volumetric_clouds = script_parameters["render_clouds"]


def save_scene_parameters(output_filename="assets/scripts/scene_parameters.json"):
        environment = Main.scene.GetEnvironment()
        script_parameters = {"sunlight_color": color_to_list(Main.ligth_sun.GetLight().GetDiffuseColor()),
                "skylight_color": color_to_list(Main.ligth_sky.GetLight().GetDiffuseColor()),
                "ambient_color": color_to_list(environment.GetAmbientColor()),
                "ambient_intensity": environment.GetAmbientIntensity(), "render_clouds": Main.render_volumetric_clouds
                                                 }
        json_script = json.dumps(script_parameters, indent=4)
        return hg.GetFilesystem().StringToFile(output_filename, json_script)


def load_game_parameters(file_name="assets/scripts/dogfight.json"):
        json_script = hg.GetFilesystem().FileToString(file_name)
        if json_script != "":
                script_parameters = json.loads(json_script)
                set_p1_missiles_smoke_color(list_to_color(script_parameters["p1_missiles_smoke_color"]))
                set_p2_missiles_smoke_color(list_to_color(script_parameters["p2_missiles_smoke_color"]))
                set_p1_gun_color(list_to_color(script_parameters["p1_gun_color"]))
                set_p2_gun_color(list_to_color(script_parameters["p2_gun_color"]))
                Main.radial_blur_strength=script_parameters["radial_blur_strength"]
                Main.deceleration_blur_strength=script_parameters["deceleration_blur_strength"]
                Main.acceleration_blur_strength=script_parameters["acceleration_blur_strength"]

def save_game_parameters(output_filename="assets/scripts/dogfight.json"):
        script_parameters = {"p1_missiles_smoke_color": color_to_list(Main.p1_missiles_smoke_color),
                "p2_missiles_smoke_color": color_to_list(Main.p2_missiles_smoke_color),
                "p1_gun_color": color_to_list(Main.bullets.colors[0]),
                "p2_gun_color": color_to_list(Main.ennemy_bullets.colors[0]),
                "radial_blur_strength": Main.radial_blur_strength,
                "deceleration_blur_strength": Main.deceleration_blur_strength,
                "acceleration_blur_strength": Main.acceleration_blur_strength
                                                 }
        json_script = json.dumps(script_parameters, indent=4)
        return hg.GetFilesystem().StringToFile(output_filename, json_script)


def post_processes_save_parameters(output_filename="assets/scripts/post_render.json"):
        script_parameters = {"hue": Main.HSL_postProcess.GetH(),
                                                 "saturation": Main.HSL_postProcess.GetS(),
                                                 "value":Main.HSL_postProcess.GetL(),
                                                 "flag_MotionBlur":Main.flag_MotionBlur,
                                                 "mb_blur_radius":Main.MotionBlur_postProcess.GetBlurRadius(),
                                                 "mb_exposure":Main.MotionBlur_postProcess.GetExposure(),
                                                 "mb_sample_count":Main.MotionBlur_postProcess.GetSampleCount()
                                                 }
        json_script = json.dumps(script_parameters, indent=4)
        return hg.GetFilesystem().StringToFile(output_filename, json_script)


def post_processes_load_parameters(file_name="assets/scripts/post_render.json"):
        json_script = hg.GetFilesystem().FileToString(file_name)
        if json_script != "":
                script_parameters = json.loads(json_script)
                Main.HSL_postProcess.SetH(script_parameters["hue"])
                Main.HSL_postProcess.SetS(script_parameters["saturation"])
                Main.HSL_postProcess.SetL(script_parameters["value"])

                Main.flag_MotionBlur=script_parameters["flag_MotionBlur"]
                Main.MotionBlur_postProcess.SetBlurRadius(script_parameters["mb_blur_radius"])
                Main.MotionBlur_postProcess.SetExposure(script_parameters["mb_exposure"])
                Main.MotionBlur_postProcess.SetSampleCount(int(script_parameters["mb_sample_count"]))

                Main.camera.RemoveComponent(Main.MotionBlur_postProcess)
                if Main.flag_MotionBlur:
                        Main.camera.AddComponent(Main.MotionBlur_postProcess)

def load_fps_matrix(fps):
        pos, rot = load_json_matrix("assets/scripts/camera_position.json")
        if pos is not None and rot is not None:
                fps.Reset(pos, rot)
def init_scene(plus):
        Main.scene = plus.NewScene()
        Main.camera = plus.AddCamera(Main.scene, hg.Matrix4.TranslationMatrix(hg.Vector3(0, 10, -10)))

        Main.camera.SetName("Camera")
        Main.camera.GetCamera().SetZNear(1.)
        Main.camera.GetCamera().SetZFar(40000)

        plus.LoadScene(Main.scene, "assets/aircraft/aircraft.scn")
        plus.LoadScene(Main.scene, "assets/ennemyaircraft/ennemy_aircraft.scn")
        plus.LoadScene(Main.scene, "assets/aircraft_carrier/aircraft_carrier.scn")
        plus.LoadScene(Main.scene, "assets/island/island.scn")
        plus.LoadScene(Main.scene, "assets/feed_backs/feed_backs.scn")

        init_lights(plus)

        while not Main.scene.IsReady():  # Wait until scene is ready
                #plus.UpdateScene(Main.scene, plus.UpdateClock())
                Main.scene.Commit()
                Main.scene.WaitCommit()

        #for i in range(256):
        #   plus.UpdateScene(Main.scene, plus.UpdateClock())
        #       Main.scene.Commit()
        #       Main.scene.WaitCommit()

        Main.satellite_camera = plus.AddCamera(Main.scene, hg.Matrix4.TranslationMatrix(hg.Vector3(0, 1000, 0)))
        setup_satellite_camera(Main.satellite_camera)

        # ---- Clouds:
        json_script = hg.GetFilesystem().FileToString("assets/scripts/clouds_parameters.json")
        if json_script != "":
                clouds_parameters = json.loads(json_script)
                Main.clouds = Clouds(plus, Main.scene, Main.scene.GetNode("Sun"), Main.resolution, clouds_parameters)

        Main.island = Main.scene.GetNode("island")
        Main.island.GetTransform().SetPosition(hg.Vector3(0, 0, 3000))
        Main.island.GetTransform().SetRotation(hg.Vector3(0, 0, 0))

        Main.sea_render_script = hg.RenderScript("assets/lua_scripts/sea_render.lua")
        Main.sea_render_script.SetEnabled(False)
        Main.sea_render = SeaRender(plus, Main.scene, Main.sea_render_script)
        Main.sea_render.load_json_script()

        Main.sea_render.update_render_script(Main.scene, Main.resolution, hg.time_to_sec_f(plus.GetClock()))
        Main.scene.AddComponent(Main.sea_render_script)

        Main.water_reflection = WaterReflection(plus, Main.scene, Main.resolution, Main.resolution.x / 4)

        #Main.clouds_render_script=hg.LogicScript("assets/lua_scripts/clouds_render.lua")
        #Main.scene.AddComponent(Main.clouds_render_script)

        #plus.UpdateScene(Main.scene)
        Main.scene.Commit()
        Main.scene.WaitCommit()
        load_scene_parameters()


def init_lights(plus):
        # Main light:
        Main.ligth_sun = plus.AddLight(Main.scene, hg.Matrix4.RotationMatrix(hg.Vector3(radians(25), radians(-45), 0)),
                                                                   hg.LightModelLinear)
        Main.ligth_sun.SetName("Sun")
        Main.ligth_sun.GetLight().SetDiffuseColor(hg.Color(255. / 255., 255. / 255., 255. / 255., 1.))

        Main.ligth_sun.GetLight().SetShadow(hg.LightShadowMap)  # Active les ombres portées
        Main.ligth_sun.GetLight().SetShadowRange(100)

        Main.ligth_sun.GetLight().SetDiffuseIntensity(1.)
        Main.ligth_sun.GetLight().SetSpecularIntensity(1.)

        # Sky ligth:
        Main.ligth_sky = plus.AddLight(Main.scene, hg.Matrix4.RotationMatrix(hg.Vector3(radians(54), radians(135), 0)),
                                                                   hg.LightModelLinear)
        Main.ligth_sky.SetName("SkyLigth")
        Main.ligth_sky.GetLight().SetDiffuseColor(hg.Color(103. / 255., 157. / 255., 141. / 255., 1.))
        Main.ligth_sky.GetLight().SetDiffuseIntensity(0.5)

        # Ambient:
        environment = hg.Environment()
        environment.SetAmbientColor(hg.Color(103. / 255., 157. / 255., 141. / 255., 1.))
        environment.SetAmbientIntensity(0.5)
        Main.scene.AddComponent(environment)


def find_controller(name):
        nl=name.lower()
        devices = hg.GetInputSystem().GetDevices()
        s = devices.size()
        for i in range(0, s):
                device_id = devices.at(i)
                d=device_id.lower()
                if d == nl:
                        return hg.GetInputSystem().GetDevice(device_id)
        return None
def gui_interface_scene(scene, fps):
        camera = scene.GetNode("Camera")

        l1_color = Main.ligth_sun.GetLight().GetDiffuseColor()
        l2_color = Main.ligth_sky.GetLight().GetDiffuseColor()
        environment = scene.GetEnvironment()
        amb_color = environment.GetAmbientColor()
        amb_intensity = environment.GetAmbientIntensity()

        if hg.ImGuiBegin("Scene Settings"):
                if hg.ImGuiButton("Load scene parameters"):
                        load_scene_parameters()
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save scene parameters"):
                        save_scene_parameters()

                d, f = hg.ImGuiCheckbox("Display collisions shapes", Main.show_debug_displays)
                if d:
                        Main.show_debug_displays = f
                        scene.GetPhysicSystem().SetDebugVisuals(Main.show_debug_displays)


                d, f = hg.ImGuiCheckbox("Volumetric clouds", Main.render_volumetric_clouds)
                if d:
                        Main.render_volumetric_clouds = f
                        if not f:
                                Main.clouds.clear_particles()

                pos = camera.GetTransform().GetPosition()
                hg.ImGuiText("Camera X " + str(pos.x))
                hg.ImGuiText("Camera Y " + str(pos.y))
                hg.ImGuiText("Camera Z " + str(pos.z))
                if hg.ImGuiButton("Load camera"):
                        # load_fps_matrix(fps)
                        pos, rot = load_json_matrix("assets/scripts/camera_position.json")
                        camera.GetTransform().SetPosition(pos)
                        camera.GetTransform().SetRotation(rot)
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save camera"):
                        save_json_matrix(camera.GetTransform().GetPosition(), camera.GetTransform().GetRotation(),
                                                         "assets/scripts/camera_position.json")

                if hg.ImGuiButton("Load aircraft matrix"):
                        pos, rot = load_json_matrix("assets/scripts/aircraft_position.json")
                        Main.p1_aircraft.reset(pos, rot)
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save aircraft matrix"):
                        nd = Main.p1_aircraft.get_parent_node()
                        save_json_matrix(nd.GetTransform().GetPosition(), nd.GetTransform().GetRotation(),
                                                         "assets/scripts/aircraft_position.json")

                f, c = hg.ImGuiColorEdit("Ambient color", amb_color)
                if f:
                        amb_color = hg.Color(c)
                        environment.SetAmbientColor(amb_color)
                d, f = hg.ImGuiSliderFloat("Ambient intensity", amb_intensity, 0, 1)
                if d:
                        amb_intensity = f
                        environment.SetAmbientIntensity(amb_intensity)

                f, c = hg.ImGuiColorEdit("Sunlight color", l1_color)
                if f:
                        l1_color = hg.Color(c)
                        Main.ligth_sun.GetLight().SetDiffuseColor(l1_color)

                f, c2 = hg.ImGuiColorEdit("Skylight color", l2_color)
                if f:
                        l2_color = hg.Color(c2)
                        Main.ligth_sky.GetLight().SetDiffuseColor(l2_color)
        hg.ImGuiEnd()


def gui_interface_game(scene):
        if hg.ImGuiBegin("Game Settings"):
                if hg.ImGuiButton("Load game parameters"):
                        load_game_parameters()
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save game parameters"):
                        save_game_parameters()

                f, c = hg.ImGuiColorEdit("P1 Missiles smoke color", Main.p1_missiles_smoke_color,
                                                                 hg.ImGuiColorEditFlags_NoAlpha)
                if f: set_p1_missiles_smoke_color(c)

                f, c = hg.ImGuiColorEdit("P2 Missiles smoke color", Main.p2_missiles_smoke_color)
                if f: set_p2_missiles_smoke_color(c)

                f, c = hg.ImGuiColorEdit("P1 gun color", Main.bullets.colors[0])
                if f: set_p1_gun_color(c)

                f, c = hg.ImGuiColorEdit("P2 gun color", Main.ennemy_bullets.colors[0])
                if f: set_p2_gun_color(c)

                hg.ImGuiSeparator()
                d, f = hg.ImGuiSliderFloat("Radial blur strength", Main.radial_blur_strength, 0, 1)
                if d: Main.radial_blur_strength = f
                d, f = hg.ImGuiSliderFloat("Deceleration blur strength", Main.deceleration_blur_strength, 0, 1)
                if d: Main.deceleration_blur_strength = f
                d, f = hg.ImGuiSliderFloat("Acceleration blur strength", Main.acceleration_blur_strength, 0, 1)
                if d: Main.acceleration_blur_strength = f


        hg.ImGuiEnd()


def gui_interface_sea_render(sea_render: SeaRender, scene, fps):
        if hg.ImGuiBegin("Sea & Sky render Settings"):

                if hg.ImGuiButton("Load sea parameters"):
                        sea_render.load_json_script()
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save sea parameters"):
                        sea_render.save_json_script()

                d, f = hg.ImGuiCheckbox("Water reflection", sea_render.render_scene_reflection)
                if d:
                        sea_render.render_scene_reflection = f

                d, f = hg.ImGuiSliderFloat("Texture North intensity", sea_render.tex_sky_N_intensity, 0, 1)
                if d: sea_render.tex_sky_N_intensity = f
                d, f = hg.ImGuiSliderFloat("Zenith falloff", sea_render.zenith_falloff, 1, 100)
                if d: sea_render.zenith_falloff = f

                f, c = hg.ImGuiColorEdit("Zenith color", sea_render.zenith_color)
                if f: sea_render.zenith_color = c
                f, c = hg.ImGuiColorEdit("Horizon color", sea_render.horizon_N_color)
                if f: sea_render.horizon_N_color = c

                f, c = hg.ImGuiColorEdit("Water color", sea_render.sea_color)
                if f: sea_render.sea_color = c
                f, c = hg.ImGuiColorEdit("Horizon Water color", sea_render.horizon_S_color)
                if f: sea_render.horizon_S_color = c
                f, c = hg.ImGuiColorEdit("Horizon line color", sea_render.horizon_line_color)
                if f: sea_render.horizon_line_color = c

                hg.ImGuiText("3/4 horizon line size: " + str(sea_render.horizon_line_size))

                f, d = hg.ImGuiCheckbox("Sea texture filtering", bool(sea_render.sea_filtering))
                if f:
                        sea_render.sea_filtering = int(d)
                hg.ImGuiText("5/6 max filter samples: " + str(sea_render.max_filter_samples))
                hg.ImGuiText("7/8 filter precision: " + str(sea_render.filter_precision))

                hg.ImGuiText("A/Q sea scale x: " + str(sea_render.sea_scale.x))
                hg.ImGuiText("Z/S sea scale y: " + str(sea_render.sea_scale.y))
                hg.ImGuiText("E/D sea scale z: " + str(sea_render.sea_scale.z))

                d, f = hg.ImGuiSliderFloat("Sea reflection", sea_render.sea_reflection, 0, 1)
                if d: sea_render.sea_reflection = f
                d, f = hg.ImGuiSliderFloat("Reflect offset", Main.sea_render.reflect_offset, 1, 1000)
                if d: Main.sea_render.reflect_offset = f

                hg.ImGuiSeparator()

        hg.ImGuiEnd()


link_altitudes = True
link_morphs = True
clouds_altitude = 1000
clouds_morph_level = 0.1


def gui_clouds(scene: hg.Scene, cloud: Clouds):
        global link_altitudes, link_morphs, clouds_altitude, clouds_morph_level

        if hg.ImGuiBegin("Clouds Settings"):
                if hg.ImGuiButton("Load clouds parameters"):
                        cloud.load_json_script()  # fps.Reset(cloud.cam_pos,hg.Vector3(0,0,0))
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save clouds parameters"):
                        cloud.save_json_script(scene)

                hg.ImGuiSeparator()

                hg.ImGuiText("Map position: X=" + str(cloud.map_position.x))
                hg.ImGuiText("Map position: Y=" + str(cloud.map_position.y))

                """
                d, f = hg.ImGuiSliderFloat("Far Clouds scale x", sky_render.clouds_scale.x, 100, 10000)
                if d:
                        sky_render.clouds_scale.x = f

                d, f = hg.ImGuiSliderFloat("Far Clouds scale y", sky_render.clouds_scale.y, 0, 1)
                if d:
                        sky_render.clouds_scale.y = f

                d, f = hg.ImGuiSliderFloat("Far Clouds scale z", sky_render.clouds_scale.z, 100, 10000)
                if d:
                        sky_render.clouds_scale.z = f


                d, f = hg.ImGuiSliderFloat("Far Clouds absorption", sky_render.clouds_absorption, 0, 1)
                if d:
                        sky_render.clouds_absorption = f

                """

                d, f = hg.ImGuiSliderFloat("Clouds scale x", cloud.map_scale.x, 100, 10000)
                if d:
                        cloud.set_map_scale_x(f)
                d, f = hg.ImGuiSliderFloat("Clouds scale z", cloud.map_scale.y, 100, 10000)
                if d:
                        cloud.set_map_scale_z(f)

                d, f = hg.ImGuiSliderFloat("Wind speed x", cloud.v_wind.x, -1000, 1000)
                if d:
                        cloud.v_wind.x = f

                d, f = hg.ImGuiSliderFloat("Wind speed z", cloud.v_wind.y, -1000, 1000)
                if d:
                        cloud.v_wind.y = f

                d, f = hg.ImGuiCheckbox("Link layers altitudes", link_altitudes)
                if d: link_altitudes = f
                d, f = hg.ImGuiCheckbox("Link layers morph levels", link_morphs)
                if d: link_morphs = f

                d, f = hg.ImGuiSliderFloat("Clouds altitude", clouds_altitude, 100, 10000)
                if d:
                        clouds_altitude = f
                        if link_altitudes:
                                for layer in cloud.layers:
                                        layer.set_altitude(f)

                d, f = hg.ImGuiSliderFloat("Clouds morph level", clouds_morph_level, 0, 1)
                if d:
                        clouds_morph_level = f
                        if link_morphs:
                                for layer in cloud.layers:
                                        layer.morph_level = f

                for layer in cloud.layers:
                        hg.ImGuiSeparator()
                        gui_layer(layer)

        hg.ImGuiEnd()


def gui_layer(layer: CloudsLayer):
        nm = layer.name
        hg.ImGuiText(layer.name)

        d, f = hg.ImGuiSliderFloat(nm + " particles rotation speed", layer.particles_rot_speed, -10, 10)
        if d:
                layer.set_particles_rot_speed(f)

        d, f = hg.ImGuiSliderFloat(nm + " particles morph level", layer.morph_level, -1, 1)
        if d:
                layer.morph_level = f

        d, f = hg.ImGuiSliderFloat(nm + " Absorption factor", layer.absorption * 100, 0.01, 10)
        if d:
                layer.set_absorption(f / 100)

        d, f = hg.ImGuiSliderFloat(nm + " Altitude floor", layer.altitude_floor, -2, 2)
        if d: layer.set_altitude_floor(f)

        d, f = hg.ImGuiSliderFloat(nm + " Altitude", layer.altitude, 0, 10000)
        if d: layer.set_altitude(f)

        d, f = hg.ImGuiSliderFloat(nm + " Altitude falloff", layer.altitude_falloff, 0.1, 100)
        if d: layer.set_altitude_falloff(f)

        d, f = hg.ImGuiSliderFloat(nm + " Particles min scale", layer.particles_scale_range.x, 1, 5000)
        if d:
                layer.set_particles_min_scale(f)
        d, f = hg.ImGuiSliderFloat(nm + " Particles max scale", layer.particles_scale_range.y, 1, 5000)
        if d:
                layer.set_particles_max_scale(f)
        d, f = hg.ImGuiSliderFloat(nm + " Alpha threshold", layer.alpha_threshold, 0, 1)
        if d:
                layer.alpha_threshold = f

        d, f = hg.ImGuiSliderFloat(nm + " Scale falloff", layer.scale_falloff, 1, 10)
        if d:
                layer.scale_falloff = f

        d, f = hg.ImGuiSliderFloat(nm + " Alpha scale falloff", layer.alpha_scale_falloff, 1, 10)
        if d:
                layer.alpha_scale_falloff = f

        d, f = hg.ImGuiSliderFloat(nm + " Perturbation", layer.perturbation, 0, 200)
        if d:
                layer.perturbation = f
        d, f = hg.ImGuiSliderFloat(nm + " Tile size", layer.tile_size, 1, 500)
        if d:
                layer.tile_size = f
        d, f = hg.ImGuiSliderFloat(nm + " Distance min", layer.distance_min, 0, 5000)
        if d:
                layer.set_distance_min(f)
        d, f = hg.ImGuiSliderFloat(nm + " Distance max", layer.distance_max, 100, 5000)
        if d:
                layer.set_distance_max(f)

        d, f = hg.ImGuiSliderFloat(nm + " Margin", layer.margin, 0.5, 2)
        if d:
                layer.margin = f
        d, f = hg.ImGuiSliderFloat(nm + " Focal margin", layer.focal_margin, 0.5, 2)
        if d:
                layer.focal_margin = f


def gui_post_rendering():
        if hg.ImGuiBegin("Post-rendering Settings"):
                if hg.ImGuiButton("Load post-render settings"):
                        post_processes_load_parameters()
                hg.ImGuiSameLine()
                if hg.ImGuiButton("Save post-render settings"):
                        post_processes_save_parameters()

                hg.ImGuiSeparator()

                d, f = hg.ImGuiSliderFloat("Hue", Main.HSL_postProcess.GetH(), -1, 1)
                if d: Main.HSL_postProcess.SetH(f)
                d, f = hg.ImGuiSliderFloat("Saturation", Main.HSL_postProcess.GetS(), 0, 1)
                if d: Main.HSL_postProcess.SetS(f)
                d, f = hg.ImGuiSliderFloat("Luminance", Main.HSL_postProcess.GetL(), 0, 1)
                if d: Main.HSL_postProcess.SetL(f)

                hg.ImGuiSeparator()

                d, f = hg.ImGuiCheckbox("Motion Blur", Main.flag_MotionBlur)
                if d:
                        Main.flag_MotionBlur=f
                        if f:
                                Main.camera.AddComponent(Main.MotionBlur_postProcess)
                        else:
                                Main.camera.RemoveComponent(Main.MotionBlur_postProcess)

                if Main.flag_MotionBlur:
                        pp=Main.MotionBlur_postProcess
                        d, i = hg.ImGuiSliderInt("Blur Radius", pp.GetBlurRadius(), 1, 100)
                        if d: pp.SetBlurRadius(i)
                        d, f = hg.ImGuiSliderFloat("Exposure", pp.GetExposure(), 0, 10)
                        if d : pp.SetExposure(f)
                        d, f = hg.ImGuiSliderInt("SampleCount", pp.GetSampleCount(), 1, 100)
                        if d : pp.SetSampleCount(f)

        hg.ImGuiEnd()


def gui_device_outputs(dev):
        if hg.ImGuiBegin("Paddle inputs"):
                for i in range(hg.Button0, hg.ButtonLast):
                        if dev.IsButtonDown(i):
                                hg.ImGuiText("Button" + str(i) + " pressed")

                hg.ImGuiText("InputAxisX: " + str(dev.GetValue(hg.InputAxisX)))
                hg.ImGuiText("InputAxisY: " + str(dev.GetValue(hg.InputAxisY)))
                hg.ImGuiText("InputAxisZ: " + str(dev.GetValue(hg.InputAxisZ)))
                hg.ImGuiText("InputAxisS: " + str(dev.GetValue(hg.InputAxisS)))
                hg.ImGuiText("InputAxisT: " + str(dev.GetValue(hg.InputAxisT)))
                hg.ImGuiText("InputAxisR: " + str(dev.GetValue(hg.InputAxisR)))
                hg.ImGuiText("InputRotX: " + str(dev.GetValue(hg.InputRotX)))
                hg.ImGuiText("InputRotY: " + str(dev.GetValue(hg.InputRotY)))
                hg.ImGuiText("InputRotZ: " + str(dev.GetValue(hg.InputRotZ)))
                hg.ImGuiText("InputRotS: " + str(dev.GetValue(hg.InputRotS)))
                hg.ImGuiText("InputRotT: " + str(dev.GetValue(hg.InputRotT)))
                hg.ImGuiText("InputRotR: " + str(dev.GetValue(hg.InputRotR)))
                hg.ImGuiText("InputButton0: " + str(dev.GetValue(hg.InputButton0)))
                hg.ImGuiText("InputButton1: " + str(dev.GetValue(hg.InputButton1)))
                hg.ImGuiText("InputButton2: " + str(dev.GetValue(hg.InputButton2)))
                hg.ImGuiText("InputButton3: " + str(dev.GetValue(hg.InputButton3)))
                hg.ImGuiText("InputButton4: " + str(dev.GetValue(hg.InputButton4)))
                hg.ImGuiText("InputButton5: " + str(dev.GetValue(hg.InputButton5)))
                hg.ImGuiText("InputButton6: " + str(dev.GetValue(hg.InputButton6)))
                hg.ImGuiText("InputButton7: " + str(dev.GetValue(hg.InputButton7)))
                hg.ImGuiText("InputButton8: " + str(dev.GetValue(hg.InputButton8)))
                hg.ImGuiText("InputButton9: " + str(dev.GetValue(hg.InputButton9)))
                hg.ImGuiText("InputButton10: " + str(dev.GetValue(hg.InputButton10)))
                hg.ImGuiText("InputButton11: " + str(dev.GetValue(hg.InputButton11)))
                hg.ImGuiText("InputButton12: " + str(dev.GetValue(hg.InputButton12)))
                hg.ImGuiText("InputButton13: " + str(dev.GetValue(hg.InputButton13)))
                hg.ImGuiText("InputButton14: " + str(dev.GetValue(hg.InputButton14)))
                hg.ImGuiText("InputButton15: " + str(dev.GetValue(hg.InputButton15)))
        hg.ImGuiEnd()
def animations(dts):
        pass
def autopilot_controller(aircraft: Aircraft):
        if hg.ImGuiBegin("Auto pilot"):
                f, d = hg.ImGuiCheckbox("Autopilot", aircraft.autopilot_activated)
                if f:
                        aircraft.autopilot_activated = d

                f, d = hg.ImGuiCheckbox("IA", aircraft.IA_activated)
                if f:
                        aircraft.IA_activated = d

                d, f = hg.ImGuiSliderFloat("Pitch", aircraft.autopilot_pitch_attitude, -180, 180)
                if d: aircraft.set_autopilot_pitch_attitude(f)
                d, f = hg.ImGuiSliderFloat("Roll", aircraft.autopilot_roll_attitude, -180, 180)
                if d: aircraft.set_autopilot_roll_attitude(f)
                d, f = hg.ImGuiSliderFloat("Cap", aircraft.autopilot_cap, 0, 360)
                if d: aircraft.set_autopilot_cap(f)
                d, f = hg.ImGuiSliderFloat("Altitude", aircraft.autopilot_altitude, 0, 10000)
                if d: aircraft.set_autopilot_altitude(f)
        hg.ImGuiEnd()


def control_aircraft_paddle(dts, aircraft: Aircraft):

        if (Main.controller is None) or not get_connected()[0]: return False,False,False

        ct = Main.controller
        aircraft.set_thrust_level(get_trigger_values(get_state(0))[1])
        aircraft.set_brake_level(get_trigger_values(get_state(0))[0])
        if aircraft.thrust_level>=1:
                aircraft.activate_post_combution()
        else:
                aircraft.deactivate_post_combution()
        p, y, r = True, True, True
        v = ct.GetValue(hg.InputAxisY)
        if v != 0:
                p = False
        aircraft.set_pitch_level(v)
        v = ct.GetValue(hg.InputAxisX)
        if v != 0:
                r = False
        aircraft.set_roll_level(-v)
        v = ct.GetValue(hg.InputAxisS)
        if v != 0:
                y = False
        aircraft.set_yaw_level(v)
        if ct.IsButtonDown(hg.Button0):
                aircraft.fire_gun_machine()
        elif aircraft.is_gun_activated() and not plus.KeyDown(hg.KeyEnter):
                aircraft.stop_gun_machine()

        if ct.WasButtonPressed(hg.Button2):
                aircraft.fire_missile()

        if ct.WasButtonPressed(hg.Button3):
                aircraft.next_target()

        return p, r, y


def control_aircraft_keyboard(dts, aircraft: Aircraft):
        if plus.KeyDown(hg.KeyLCtrl) and not get_connected()[0]:
                if plus.KeyPress(hg.KeyA):
                        aircraft.set_thrust_level(aircraft.thrust_level + 0.01)
                if plus.KeyPress(hg.KeyZ):
                        aircraft.set_thrust_level(aircraft.thrust_level - 0.01)
        else:
                if plus.KeyDown(hg.KeyA) and not get_connected()[0]:
                        aircraft.set_thrust_level(aircraft.thrust_level + 0.01)
                if plus.KeyDown(hg.KeyZ) and not get_connected()[0]:
                        aircraft.set_thrust_level(aircraft.thrust_level - 0.01)

        if plus.KeyDown(hg.KeyB) and not get_connected()[0]:
                aircraft.set_brake_level(1)
        elif not get_connected()[0]:
                aircraft.set_brake_level(0)
        if plus.KeyDown(hg.KeyC) and not get_connected()[0]:
                aircraft.set_flaps_level(aircraft.flaps_level + 0.01)
        if plus.KeyDown(hg.KeyV) and not get_connected()[0]:
                aircraft.set_flaps_level(aircraft.flaps_level - 0.01)

        p, y, r = True, True, True
        if plus.KeyDown(hg.KeyLeft) and not get_connected()[0]:
                aircraft.set_roll_level(1)
                r = False
        elif plus.KeyDown(hg.KeyRight) and not get_connected()[0]:
                aircraft.set_roll_level(-1)
                r = False

        if plus.KeyDown(hg.KeyUp) and not get_connected()[0]:
                aircraft.set_pitch_level(1)
                p = False
        elif plus.KeyDown(hg.KeyDown) and not get_connected()[0]:
                aircraft.set_pitch_level(-1)
                p = False

        if plus.KeyDown(hg.KeySuppr) and not get_connected()[0]:
                aircraft.set_yaw_level(-1)
                y = False
        elif plus.KeyDown(hg.KeyPageDown) and not get_connected()[0]:
                aircraft.set_yaw_level(1)
                y = False

        # aircraft.stabilize(dts, p, y, r)
        if aircraft.thrust_level>=1:
                aircraft.activate_post_combution()
        else:
                aircraft.deactivate_post_combution()
        if plus.KeyDown(hg.KeyEnter) and not get_connected()[0]:
                aircraft.fire_gun_machine()
        elif aircraft.is_gun_activated():
                aircraft.stop_gun_machine()

        if plus.KeyPress(hg.KeyF1) and not get_connected()[0]:
                aircraft.fire_missile()

        if aircraft == Main.p1_aircraft:
                if plus.KeyPress(hg.KeyT) and not get_connected()[0]:
                        aircraft.next_target()

                if plus.KeyDown(hg.KeyP) and not get_connected()[0]:
                        aircraft.set_health_level(aircraft.health_level + 0.01)
                if plus.KeyDown(hg.KeyM) and not get_connected()[0]:
                        aircraft.set_health_level(aircraft.health_level - 0.01)
        return p, r, y

def set_view(view):
        Main.current_view=view
        set_track_view(view)

def control_views():
        quit_sv = False
        if plus.KeyDown(hg.KeyNumpad2):
                quit_sv = True
                set_view(back_view)
        elif plus.KeyDown(hg.KeyNumpad8):
                quit_sv = True
                set_view(front_view)
        elif plus.KeyDown(hg.KeyNumpad4):
                quit_sv = True
                set_view(left_view)
        elif plus.KeyDown(hg.KeyNumpad6):
                quit_sv = True
                set_view(right_view)
        elif plus.KeyPress(hg.KeyNumpad5):
                if Main.satellite_view:
                        Main.satellite_view = False
                        Main.scene.SetCurrentCamera(Main.camera)
                else:
                        Main.satellite_view = True
                        Main.scene.SetCurrentCamera(Main.satellite_camera)

        if quit_sv and Main.satellite_view:
                Main.satellite_view = False
                Main.scene.SetCurrentCamera(Main.camera)

        if Main.satellite_view:
                if plus.KeyDown(hg.KeyInsert):
                        increment_satellite_view_size()
                elif plus.KeyDown(hg.KeyPageUp):
                        decrement_satellite_view_size()
def renderScript_flow(plus, t,dts):
        Main.sea_render_script.SetEnabled(True)
        if Main.sea_render.render_scene_reflection and not Main.satellite_view:
                Main.water_reflection.render(plus, Main.scene, Main.camera)

        Main.sea_render.reflect_map = Main.water_reflection.render_texture
        Main.sea_render.reflect_map_depth = Main.water_reflection.render_depth_texture
        Main.sea_render.update_render_script(Main.scene, Main.resolution, hg.time_to_sec_f(plus.GetClock()))
        # Main.scene.Commit()
        # Main.scene.WaitCommit()
        renderer = plus.GetRenderer()
        renderer.EnableDepthTest(True)
        renderer.EnableDepthWrite(True)
        renderer.EnableBlending(True)
        renderer = plus.GetRenderer()
        renderer.ClearRenderTarget()

        # Volumetric clouds:
        if Main.render_volumetric_clouds:
                Main.clouds.update(t,dts, Main.scene, Main.resolution)
                Main.scene.Commit()
                Main.scene.WaitCommit()

        plus.UpdateScene(Main.scene)

def render_flow(plus,delta_t):
        t = hg.time_to_sec_f(plus.GetClock())
        dts=hg.time_to_sec_f(delta_t)

        renderScript_flow(plus, t,dts)
def start_music(filename):
        music = Main.audio.LoadSound(filename)
        Main.title_music_settings = hg.MixerChannelState()
        Main.title_music_settings.loop_mode = hg.MixerRepeat
        Main.title_music_settings.volume = 1
        Main.title_music_settings.pitch = 1
        Main.title_music = Main.audio.Start(music, Main.title_music_settings)


def start_title_music():
        pass


def start_success_music():
        pass


def start_gameover_music():
        pass


def stop_music():
        Main.audio.Stop(Main.title_music)
def init_start_phase():
        Main.p1_success = False
        Main.audio.StopAll()
        Main.p1_sfx.stop_engine(Main)
        Main.p2_sfx.stop_engine(Main)

        load_fps_matrix(Main.fps)
        # ...or Camera
        camera = Main.scene.GetNode("Camera")
        pos, rot = load_json_matrix("assets/scripts/camera_position.json")
        camera.GetTransform().SetPosition(pos)
        camera.GetTransform().SetRotation(rot)

        # pos, rot = load_json_matrix("assets/scripts/aircraft_position.json")
        pos, rot = Main.carrier.get_aircraft_start_point()
        pos.y += 2
        Main.p1_aircraft.reset(pos, rot)
        Main.p1_aircraft.IA_activated = False

        # On aircraft carrier:
        # pos.z += 40
        # pos.x -= 5
        # Main.p2_aircraft.reset(pos, rot)

        # On flight:

        Main.p2_aircraft.reset(hg.Vector3(uniform(10000, -10000), uniform(1000, 7000), uniform(10000, -10000)),
                                                   hg.Vector3(0, radians(uniform(-180, 180)), 0))
        Main.p2_aircraft.set_thrust_level(0)

        Main.p1_sfx.reset()
        Main.p2_sfx.reset()

        #plus.UpdateScene(Main.scene)
        Main.scene.Commit()
        Main.scene.WaitCommit()

        setup_camera_follow(Main.p1_aircraft.get_parent_node(),
                                                Main.p1_aircraft.get_parent_node().GetTransform().GetPosition(),
                                                Main.p1_aircraft.get_parent_node().GetTransform().GetWorld().GetRotationMatrix())

        Main.scene.SetCurrentCamera(Main.camera)
        Main.satellite_view = False

        Main.HSL_postProcess.SetS(0)
        Main.HSL_postProcess.SetL(0)


        #plus.UpdateScene(Main.scene)
        Main.scene.Commit()
        Main.scene.WaitCommit()
        Main.fading_cptr = 0

        plus.SetFont(Main.title_font)
        for i in range(4):
                Main.p1_aircraft.fit_missile(Main.p1_missiles[i], i)
                Main.p2_aircraft.fit_missile(Main.p2_missiles[i], i)

        Main.audio.SetMasterVolume(1)
        start_title_music()

        Main.RadialBlur_postProcess.SetStrength(0)

        return start_phase


def start_phase(plus, delta_t):
        dts = hg.time_to_sec_f(delta_t)
        camera = Main.scene.GetNode("Camera")
        global pp
        # Main.fps.UpdateAndApplyToNode(camera, delta_t)
        Main.camera_matrix = update_camera_follow(camera, dts)
        Main.camera_v_move = camera_move * dts

        # Kinetics:

        animations(dts)
        Main.carrier.update_kinetics(Main.scene, dts)
        Main.p1_aircraft.update_kinetics(Main.scene, dts)
        # Main.p2_aircraft.update_kinetics(Main.scene, dts)

        # fade in:
        fade_in_delay = 1.
        Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)

        Main.HSL_postProcess.SetL(Main.fading_cptr / fade_in_delay)

        if Main.fading_cptr >= fade_in_delay:
                # Start infos:
                f = Main.HSL_postProcess.GetL()
                plus.Text2D(514 / 1600 * Main.resolution.x, 771 / 900 * Main.resolution.y, "GET READY",
                                        0.08 * Main.resolution.y, hg.Color(1., 0.9, 0.3, 1) * f)

                plus.Text2D(554 / 1600 * Main.resolution.x, 671 / 900 * Main.resolution.y,
                                        "Ennemy aircraft detected : Shoot it down !", 0.02 * Main.resolution.y,
                                        hg.Color(1., 0.9, 0.3, 1) * f, Main.hud_font)

                plus.Text2D(640 / 1600 * Main.resolution.x, 591 / 900 * Main.resolution.y, "Hit space or Start",
                                        0.025 * Main.resolution.y,
                                        hg.Color(1, 1, 1, (1 + sin(hg.time_to_sec_f(plus.GetClock() * 10))) * 0.5) * f)

                s = 0.015
                x = 470 / 1600 * Main.resolution.x
                y = 350
                c = hg.Color(1., 0.9, 0.3, 1) * f
                # Function
                plus.Text2D(x, y / 900 * Main.resolution.y, "Thrust level", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Pitch", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Roll", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "SYaw", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "Gun", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "Missiles", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "Target selection", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "Brake", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "Flaps", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "Reset game", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 200) / 900 * Main.resolution.y, "Set View", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 220) / 900 * Main.resolution.y, "Satellite Zoom", s * Main.resolution.y, c, Main.hud_font)
                # Keyboard
                c = hg.Color.White
                x = 815 / 1600 * Main.resolution.x
                plus.Text2D(x, y / 900 * Main.resolution.y, "A / Z", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Up / Down", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Right / Left", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "Suppr / Page dn", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "ENTER", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "F1", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "T", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "B", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "C / V", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "Backspace", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 200) / 900 * Main.resolution.y, "2/4/8/6/5", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 220) / 900 * Main.resolution.y, "Insert / Page Up", s * Main.resolution.y, c, Main.hud_font)

                # Paddle
                x = 990 / 1600 * Main.resolution.x
                plus.Text2D(x, y / 900 * Main.resolution.y, "RT", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 20) / 900 * Main.resolution.y, "Left Stick", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 40) / 900 * Main.resolution.y, "Left Stick", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 60) / 900 * Main.resolution.y, "Right Stick Left/Right", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 80) / 900 * Main.resolution.y, "A", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 100) / 900 * Main.resolution.y, "X", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 120) / 900 * Main.resolution.y, "Y", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 140) / 900 * Main.resolution.y, "LT", s * Main.resolution.y, c, Main.hud_font)
                plus.Text2D(x, (y - 160) / 900 * Main.resolution.y, "Cross Right / LEft", s * Main.resolution.y, c,
                                        Main.hud_font)
                plus.Text2D(x, (y - 180) / 900 * Main.resolution.y, "Back", s * Main.resolution.y, c, Main.hud_font)

                start = False
                if Main.controller is not None:
                        if Main.controller.WasButtonPressed(hg.ButtonStart): start = True
                if plus.KeyPress(hg.KeySpace) or start:
                        return init_main_phase()

        # rendering:
        render_flow(plus,delta_t)

        return start_phase


def init_main_phase():
        stop_music()

        Main.HSL_postProcess.SetL(1)
        Main.HSL_postProcess.SetS(0)

        set_view(back_view)

        Main.p1_aircraft.set_thrust_level(0.7)
        # Main.p1_aircraft.set_brake_level(1)
        # Main.p1_aircraft.activate_post_combution()

        # p2 on carrier:
        # Main.p2_aircraft.set_thrust_level(0)
        # Main.p2_aircraft.set_flaps_level(0.)
        # Main.p2_aircraft.set_brake_level(0)
        # Main.p2_aircraft.activate_post_combution()

        # p2 on flight:
        Main.p2_aircraft.set_linear_speed(800 / 3.6)
        Main.p2_aircraft.IA_activated = True

        Main.p1_aircraft.targets = [Main.p2_aircraft]
        Main.p2_aircraft.targets = [Main.p1_aircraft]
        Main.p2_aircraft.set_target_id(1)

        plus.SetFont(Main.hud_font)
        Main.fading_cptr = 0

        Main.p1_targets = [Main.p2_aircraft, Main.carrier]
        for i in range(len(Main.p1_missiles)):
                Main.p1_targets.append(Main.p1_missiles[i])
                Main.p1_targets.append(Main.p2_missiles[i])

        Main.fadout_flag = False
        Main.fadout_cptr = 0
        return main_phase


def update_radial_post_process(acceleration):
        if acceleration<0:
                Main.RadialBlur_postProcess.SetStrength(Main.radial_blur_strength*pow(min(1, abs(acceleration*Main.deceleration_blur_strength)), 2))
        else:
                Main.RadialBlur_postProcess.SetStrength(Main.radial_blur_strength*pow(min(1, abs(acceleration*Main.acceleration_blur_strength)), 4))

        if Main.gun_sight_2D is not None and Main.current_view==back_view:
                Main.RadialBlur_postProcess.SetCenter(Main.gun_sight_2D)
        else:
                Main.RadialBlur_postProcess.SetCenter(hg.Vector2(0.5,0.5))

def main_phase(plus, delta_t):
        dts = hg.time_to_sec_f(delta_t)
        camera = Main.scene.GetNode("Camera")
        
        acc=Main.p1_aircraft.get_linear_acceleration()
        noise_level = max(0, Main.p1_aircraft.get_linear_speed() * 3.6 / 2500 * 0.1 + pow(
                min(1, abs(acc / 7)), 2) * 1)
        if Main.p1_aircraft.post_combution:
                noise_level += 0.1

        if Main.satellite_view:
                update_satellite_camera(Main.satellite_camera, Main.resolution.x / Main.resolution.y, dts)
        Main.camera_matrix = update_camera_tracking(camera, dts, noise_level)
        Main.camera_v_move = camera_move * dts
        #Main.fps.UpdateAndApplyToNode(camera, delta_t)
        #Main.camera_matrix = None

        update_radial_post_process(acc)

        # Kinetics:
        fade_in_delay = 1.
        if Main.fading_cptr < fade_in_delay:
                Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)
                Main.HSL_postProcess.SetS( Main.fading_cptr / fade_in_delay * 0.75)

        animations(dts)
        Main.carrier.update_kinetics(Main.scene, dts)
        Main.p1_aircraft.update_kinetics(Main.scene, dts)
        Main.p2_aircraft.update_kinetics(Main.scene, dts)

        pk, rk, yk = control_aircraft_keyboard(dts, Main.p1_aircraft)
        if Main.controller is not None:
                pp, rp, yp = control_aircraft_paddle(dts, Main.p1_aircraft)

        else:
                pp, rp, yp = True, True, True

        Main.p1_aircraft.stabilize(dts, pk & pp, yk & yp, rk & rp)
        # Hud
        display_hud(Main, plus, Main.p1_aircraft, Main.p1_targets)

        control_views()

        # SFX:
        Main.p1_sfx.update_sfx(Main, dts)
        Main.p2_sfx.update_sfx(Main, dts)

        # rendering:
        render_flow(plus,delta_t)

        if Main.fadout_flag:
                Main.fadout_cptr += dts
                fadout_delay = 1
                f = Main.fadout_cptr / fadout_delay
                Main.audio.SetMasterVolume(1 - f)
                Main.HSL_postProcess.SetL(max(0, 1 - f))
                if Main.fadout_cptr > fadout_delay:
                        Main.HSL_postProcess.SetL(0)
                        return init_start_phase()

        back = False
        if Main.controller is not None:
                if Main.controller.WasButtonPressed(hg.ButtonBack): back = True
        if plus.KeyPress(hg.KeyBackspace) or back:
                Main.fadout_flag = True

        if Main.p1_aircraft.wreck:
                Main.p1_success = False
                return init_end_phase()

        if Main.p2_aircraft.wreck:
                Main.p1_success = True
                return init_end_phase()

        return main_phase


def init_end_phase():
        Main.fadout_flag = False
        plus.SetFont(Main.title_font)
        Main.fading_cptr = 0
        Main.p1_aircraft.IA_activated = True
        Main.scene.SetCurrentCamera(Main.camera)
        Main.satellite_view = False
        Main.fading_start_saturation = Main.HSL_postProcess.GetS()
        if Main.p1_success:
                start_success_music()

        Main.RadialBlur_postProcess.SetStrength(0)

        return end_phase


def end_phase(plus, delta_t):
        dts = hg.time_to_sec_f(delta_t)

        camera = Main.scene.GetNode("Camera")
        Main.camera_matrix = update_camera_follow(camera, dts)
        Main.camera_v_move = camera_move * dts

        # Kinetics:
        if Main.p1_success:
                msg = "MISSION SUCCESSFUL !"
                x = 435 / 1600
                fade_in_delay = 10
                s = 50 / 900
        else:
                msg = "YOU DIED"
                x = 550 / 1600
                fade_in_delay = 1
                s = 72 / 900
        if Main.fading_cptr < fade_in_delay:
                Main.fading_cptr = min(fade_in_delay, Main.fading_cptr + dts)
                Main.HSL_postProcess.SetS((1 - Main.fading_cptr / fade_in_delay) * Main.fading_start_saturation)

        animations(dts)
        Main.carrier.update_kinetics(Main.scene, dts)
        Main.p1_aircraft.update_kinetics(Main.scene, dts)
        Main.p2_aircraft.update_kinetics(Main.scene, dts)

        # Hud

        f = Main.HSL_postProcess.GetL()
        plus.Text2D(x * Main.resolution.x, 611 / 900 * Main.resolution.y, msg, s * Main.resolution.y,
                                hg.Color(1., 0.9, 0.3, 1) * f)

        plus.Text2D(645 / 1600 * Main.resolution.x, 531 / 900 * Main.resolution.y, "Hit Space or Start",
                                0.025 * Main.resolution.y,
                                hg.Color(1, 1, 1, (1 + sin(hg.time_to_sec_f(plus.GetClock() * 10))) * 0.5) * f)

        # SFX:
        Main.p1_sfx.update_sfx(Main, dts)
        Main.p2_sfx.update_sfx(Main, dts)

        # rendering:
        render_flow(plus,delta_t)

        start = False
        if Main.controller is not None:
                if Main.controller.WasButtonPressed(hg.ButtonStart): start = True

        if plus.KeyPress(hg.KeySpace) or start:
                Main.fadout_flag = True
                Main.fadout_cptr = 0

        if Main.fadout_flag:
                Main.fadout_cptr += dts
                fadout_delay = 1
                f = Main.fadout_cptr / fadout_delay
                Main.audio.SetMasterVolume(1 - f)
                Main.HSL_postProcess.SetL(max(0, 1 - f))
                if Main.fadout_cptr > fadout_delay:
                        Main.HSL_postProcess.SetL(0)
                        return init_start_phase()

        return end_phase
plus = hg.GetPlus()
hg.LoadPlugins()
hg.MountFileDriver(hg.StdFileDriver())

# hg.SetLogIsDetailed(True)
# hg.SetLogLevel(hg.LogAll)
smr_ok,scr_mod,scr_res = request_screen_mode()
if smr_ok == "ok":
        Main.resolution.x,Main.resolution.y=scr_res.x,scr_res.y
        Main.screenMode=scr_mod

        plus.RenderInit(int(Main.resolution.x), int(Main.resolution.y), Main.antialiasing, Main.screenMode)
        plus.SetBlend2D(hg.BlendAlpha)
        plus.SetBlend3D(hg.BlendAlpha)
        plus.SetCulling2D(hg.CullNever)
        plus.Clear()
        plus.Flip()
        plus.EndFrame()
        init_game(plus)
        #plus.UpdateScene(Main.scene)
        Main.scene.Commit()
        Main.scene.WaitCommit()
        game_phase = init_start_phase()

        while not plus.KeyDown(hg.KeyEscape) and not plus.IsAppEnded():
                delta_t = plus.UpdateClock()
                dts = hg.time_to_sec_f(delta_t)

                if plus.KeyPress(hg.KeyF12):
                        if Main.display_gui:
                                Main.display_gui = False
                        else:
                                Main.display_gui = True
                if Main.display_gui:
                        hg.ImGuiMouseDrawCursor(True)
                        gui_interface_sea_render(Main.sea_render, Main.scene, Main.fps)
                        gui_interface_scene(Main.scene, Main.fps)
                        gui_interface_game(Main.scene)
                        gui_post_rendering()
                        gui_clouds(Main.scene, Main.clouds)
                # edition_clavier(Main.sea_render)
                # autopilot_controller(Main.p1_aircraft)
                # Update game state:

                if Main.show_debug_displays:
                        DebugDisplays.affiche_vecteur(plus, Main.camera, Main.p1_aircraft.ground_ray_cast_pos,
                                                                                  Main.p1_aircraft.ground_ray_cast_dir * Main.p1_aircraft.ground_ray_cast_length,
                                                                                  False)

                if plus.KeyDown(hg.KeyK):
                        Main.p2_aircraft.start_explosion()
                        Main.p2_aircraft.set_health_level(0)

                # Rendering:
                game_phase = game_phase(plus, delta_t)

                # End rendering:
                plus.Flip()
                plus.EndFrame()

        plus.RenderUninit()
