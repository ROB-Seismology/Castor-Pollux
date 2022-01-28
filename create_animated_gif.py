import os
import imageio


def create_animated_gif(folder, img_basename, ext=".png", duration=1):
	images = []
	for file in os.listdir(folder):
		if (file[:len(img_basename)] == img_basename and
			os.path.splitext(file)[1] in (ext, ext.upper())):
			img_filespec = os.path.join(folder, file)
			img = imageio.imread(img_filespec)
			images.append(img)

	out_file = os.path.join(folder, img_basename + ".gif")
	imageio.mimsave(out_file, images, duration=duration, subrectangles=True)


if __name__ == "__main__":
	folder = r"E:\Home\_kris\Publications\2017 - Aysen\Figures\Sensitivity\v4"

	ipe_names = ["Barrientos1980WithSigma", "BakunWentworth1997WithSigma", "AllenEtAl2012", "AtkinsonWald2007"]

	## Resolution power figures
	for ipe_name in ipe_names:
		img_basename = "ResPow_%s" % ipe_name
		#print img_basename
		#create_animated_gif(folder, img_basename)

	## Scenario sensitivity figures
	#scenarios = ["Quitralco", "Azul Tigre South", "2007", "Due East", "Due West"]
	scenarios = ["Quitralco"]
	for scenario in scenarios:
		for mmi in ["6.5", "9"]:
			#for ipe_name in ["BakunWentworth1997WithSigma", "LogicTree"]:
			for ipe_name in ["BakunWentworth1997WithSigma"]:
				img_basename = "%s_MMI=%s_%s" % (scenario, mmi, ipe_name)
				print(img_basename)
				create_animated_gif(folder, img_basename)
