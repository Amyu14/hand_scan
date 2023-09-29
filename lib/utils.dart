import 'package:flutter/material.dart';
import 'package:handscan/screens/selected_image_screen.dart';
import 'package:image_picker/image_picker.dart';

Future pickImage(ImageSource source, BuildContext context) async {
  ImagePicker picker = ImagePicker();
  XFile? file = await picker.pickImage(source: source);
  if (file != null) {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (ctx) {
        return SelectedImageScreen(file);
      })
    );
  }
}